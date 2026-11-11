from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from attacks.base import BaseAttack, FeatureSpaceAttack, ProblemSpaceAttack
from constraints.gatekeeper import UniversalConstraintGatekeeper
from core.exceptions import ConfigurationError
from core.provenance import load_split_manifest, split_indices, validate_research_split_manifest, manifest_hash, source_revision_sha256, frozen_model_state_sha256, generator_artifact_provenance, _file_sha256
from core.registry import DETECTORS, PREPROCESSORS
from evaluator.engine import EvaluationEngine
from evaluator.logger import ExperimentReporter
from evaluator.metrics import classification_metrics, classification_support, CLASSIFICATION_METRIC_POLICY_VERSION
from models.wrapper import ModelWrapper, PyTorchModelWrapper
from orchestrator.device import configure_process_device
from orchestrator.runner import seed_everything
from orchestrator.progress import ProgressCallback
from .seeds import role_seed
from .experiment import ExperimentStore
from .factory import _build_model, _cfg, _inject_dynamic_dataset_shape, _instantiate_attack, _instantiate_dataset
from .io import _load_checkpoint, _load_checkpoint_metadata, _normalization_from_checkpoint, _save_json

def evaluate_attack_seed(*, config: Mapping[str, Any], seed: int, gpu: str,
                         logger: logging.Logger | None = None,
                         progress: ProgressCallback | None = None) -> Mapping[str, float]:
    """Evaluate one attack seed against a frozen target-model checkpoint.

    The optional surrogate is used only to generate adversarial inputs; target
    inference, constraint checks, and reporting remain owned by the target
    evaluation path.
    """
    cfg = _cfg(config)
    device_info = configure_process_device(gpu)
    attack_seed = role_seed(cfg, 'attack', seed)
    target_seed = role_seed(cfg, 'target', seed)
    surrogate_seed = role_seed(cfg, 'surrogate', seed)
    detector_seed = role_seed(cfg, 'detector', target_seed)
    seed_everything(attack_seed)
    _inject_dynamic_dataset_shape(cfg)
    if progress is not None:
        progress('load_dataset')
    dataset = _instantiate_dataset(cfg)
    x_raw, y_np = dataset.raw_arrays()
    y_np = np.asarray(y_np, dtype=np.int64)
    # CICIoMT2024 fixes benign=0 in the encoded class mapping. A nonzero
    # override must fail before even clean frozen-target inference, rather
    # than silently change which class counts as benign evasion.
    evaluation_cfg = dict(cfg.get('evaluation', {}))
    if str(cfg['dataset']['name']) == 'ciciomt2024':
        mapping = dict(getattr(dataset, 'class_mapping', {}) or {})
        mapped_benign = mapping.get(getattr(dataset, 'benign_string', 'Benign'),
                                    mapping.get('0', 0))
        if int(mapped_benign) != 0 or int(evaluation_cfg.get('benign_label', 0)) != 0:
            raise ConfigurationError(
                'CICIoMT2024 benign class is encoded as label 0; '
                'evaluation.benign_label must be 0 before target scoring'
            )
    if progress is not None:
        progress('validate_split_checkpoint')
    # Every reported attack uses target-checkpoint test membership, never the
    # entire training corpus or a newly sampled subset of it.
    store = ExperimentStore(cfg.get('experiment_dir'), dataset=cfg['dataset']['name'])
    target_cfg = dict(cfg["model"])
    target_ckpt = store.model_path(
        target_cfg['name'], target_cfg.get('training', {}).get('defense_mode', 'none'), target_seed,
    )
    store.validate_template(target_cfg.get('checkpoint'), target_ckpt, target_seed, role='model')
    store.require('model', target_ckpt, seed=target_seed)
    target_meta = _load_checkpoint_metadata(target_ckpt)
    if target_meta.get('experiment_id') != store.read()['experiment_id']:
        raise ConfigurationError('Target checkpoint belongs to another experiment version')
    target_split_seed = int((target_meta.get('seed_roles') or {}).get('split', target_seed))
    # With a fixed target checkpoint, the split is defined by that checkpoint,
    # not regenerated from the attack replicate's RNG.
    if cfg.get('seed_roles', {}).get('split') is not None and (
            role_seed(cfg, 'split', seed) != target_split_seed):
        raise ConfigurationError('Configured split seed disagrees with frozen target manifest')
    manifest = load_split_manifest(
        target_ckpt.parent / "split_manifest.json", dataset, y_np, target_split_seed,
        expected_hash=target_meta.get("split_manifest_sha256"),
    )
    if target_meta.get("split_manifest_sha256") is None:
        raise ConfigurationError("Target checkpoint lacks mandatory split provenance; retrain")
    if target_meta.get("class_mapping", {}) != getattr(dataset, "class_mapping", {}):
        raise ConfigurationError("Target class-ID semantics differ from dataset; retrain checkpoint")
    if tuple(target_meta.get("input_shape", ())) != tuple(x_raw.shape[1:]):
        raise ConfigurationError("Target checkpoint topology differs from fixed dataset transform; retrain")
    if target_meta.get("dataset_identity") != manifest["identity"]:
        raise ConfigurationError("Target checkpoint lacks matching transformation identity; retrain")
    # A checkpoint-defined manifest is authoritative. Detect a configuration
    # that *claims* another split fraction instead of silently ignoring it.
    split_config = dict(cfg.get('data_split') or {})
    for fraction_name in ('test_fraction', 'validation_fraction'):
        if (fraction_name in split_config
                and float(split_config[fraction_name]) != float(manifest['split_policy'][fraction_name])):
            raise ConfigurationError(
                f'Evaluation {fraction_name} disagrees with frozen target split manifest'
            )
    budget_profile = str(cfg.get('budget', {}).get('name', 'smoke'))
    if budget_profile not in {'smoke', 'research'}:
        raise ConfigurationError('Evaluation budget must be research or smoke')
    max_samples = evaluation_cfg.get("max_samples")
    if budget_profile == 'research':
        if max_samples is not None:
            raise ConfigurationError('Research evaluation must use the full held-out test partition: max_samples=null')
        validate_research_split_manifest(manifest, y_np, dataset)
        if target_meta.get('training_budget_profile') != 'research':
            raise ConfigurationError('Research evaluation requires a research-trained target checkpoint; retrain')
    sample_indices = split_indices(manifest, dataset, "test")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples < 1:
            raise ConfigurationError("evaluation.max_samples must be >= 1 or null")
        if len(sample_indices) > max_samples:
            sample_indices = np.sort(np.random.default_rng(
                int(evaluation_cfg.get('sample_selection_seed', target_split_seed))
            ).choice(
                sample_indices, size=max_samples, replace=False
            ))
    # A single sample selection is shared by physical, normalized, label,
    # occupancy, and raw-traffic views; avoid normalizing discarded records.
    x_raw = x_raw[sample_indices]
    y_np = y_np[sample_indices]
    feature_names = tuple(dataset.feature_names)
    if logger:
        logger.info(
            "evaluate.start seed=%d device=%s dataset=%s samples=%d features=%d attack=%s "
            "target=%s max_samples=%s",
            seed, device_info.torch_device, cfg["dataset"]["name"], len(y_np),
            len(feature_names), cfg["attack"]["name"], cfg["model"]["name"], max_samples or "all",
        )
    
    # keep x_raw for floating drift checks, but the target model will receive a normalized copy
    clean_physical = x_raw
    # -------------------- target model --------------------
    if progress is not None:
        progress('load_target_checkpoint')
    target_norm = _normalization_from_checkpoint(target_meta, feature_names, role="target model")
    x_target = target_norm.normalize(x_raw)
    # The target-normalized copy is now the evaluation input.
    # Drop the dataset plugin's canonical/raw cache before loading
    # the surrogate or starting the attack. Sequence datasets implement this
    # release hook; other dataset plugins may simply be garbage-collected here.
    release_raw_features = getattr(dataset, "release_raw_features", None)
    if callable(release_raw_features):
        release_raw_features()
    gc.collect()
    target_num_classes = int(target_meta["num_classes"])
    target_wrapper = _build_model(
        target_cfg,
        tuple(x_target.shape),
        device=device_info.torch_device,
        num_classes=target_num_classes,
        seed=target_seed,
    )
    _load_checkpoint(target_wrapper, target_ckpt, checkpoint_data=target_meta)
    target_digest = target_meta.get("model_state_sha256")
    if not isinstance(target_digest, str) or frozen_model_state_sha256(target_wrapper) != target_digest:
        raise ConfigurationError("Frozen target checkpoint lacks matching final-state fingerprint; retrain target")
    if logger:
        logger.info(
            "evaluate.target_loaded seed=%d model=%s family=%s checkpoint=%s input_shape=%s",
            seed, target_cfg["name"], type(target_wrapper).__name__, target_ckpt, tuple(x_target.shape[1:]),
        )

    # Optional inference-time detector fitted after the target model's training-time
    # defense. Detector state lives adjacent to the classifier checkpoint.
    detector_cfg = dict(cfg.get("detector", {}) or {})
    detector = None
    detector_artifact = None
    if bool(detector_cfg.get("enabled", False)):
        if not callable(getattr(target_wrapper, "extract_features", None)):
            raise ConfigurationError(
                f"Target wrapper {type(target_wrapper).__name__} does not support detector feature extraction"
            )
        detector_name = str(detector_cfg.get("name", "mahalanobis"))
        detector_params = dict(detector_cfg.get("params", {}))
        detector_cls = DETECTORS.get(detector_name)
        detector = detector_cls(**detector_params)
        if detector_cfg.get('checkpoint'):
            raise ConfigurationError('Explicit external detector checkpoints are not supported in experiment_dir')
        detector_path = store.detector_path(target_ckpt)
        if detector_seed != target_seed:
            raise ConfigurationError('Detector must be bound to the selected target seed and model state')
        store.require('detector', detector_path, seed=detector_seed)
        if not detector_path.exists():
            raise ConfigurationError(
                f"Detector {detector_name!r} is enabled but its fitted state is missing: {detector_path}. "
                "Run train_nids with detector.enabled=true after training the classifier."
            )
        detector_payload = torch.load(detector_path, map_location="cpu", weights_only=False)
        if int(detector_payload.get('seed', -1)) != detector_seed:
            raise ConfigurationError('Detector artifact seed differs from selected detector seed')
        if (detector_payload.get("split_manifest_sha256") != target_meta["split_manifest_sha256"]
                or detector_payload.get("fit_split") != "train"
                or detector_payload.get("target_model_state_sha256") != target_digest):
            raise ConfigurationError(
                "Detector checkpoint does not match this exact frozen target state and clean training split; retrain detector"
            )
        saved_name = str(detector_payload.get("name", detector_name))
        if saved_name != detector_name:
            raise ConfigurationError(
                f"Configured detector {detector_name!r} does not match checkpoint detector {saved_name!r}"
            )
        # The registry proves checkpoint identity, but not that the requested
        # fit regime matches its fitted covariance. The stored state overrides
        # constructor settings, so reconcile fit-critical parameters first.
        if detector_name == "mahalanobis":
            state = detector_payload["state_dict"]
            fit_key = "covariance_ridge"
            saved_ridge = state.get(fit_key)
            declared_ridge = (detector_payload.get("params") or {}).get(fit_key, saved_ridge)
            if (saved_ridge is None or float(declared_ridge) != float(saved_ridge)
                    or float(saved_ridge) != detector.covariance_ridge):
                raise ConfigurationError(
                    "Detector covariance_ridge differs from the registered fitted detector; "
                    "evaluate using the fitted ridge or refit on authorized clean training data"
                )
            # Scoring batch size changes only bounded inference batching, not
            # fitted means/covariance. Apply the requested setting after load.
            scoring_batch_size = detector.score_batch_size
        if detector_name == 'mahalanobis_multilayer_val_calibrated':
            state = detector_payload['state_dict']
            if (state.get('calibration_coef') is None
                    or detector_payload.get('validation_membership_sha256') != state.get('calibration_membership_sha256')
                    or state.get('covariance_ridge') != detector.covariance_ridge
                    or state.get('preprocessing_magnitude') != detector.preprocessing_magnitude):
                raise ConfigurationError('Lee detector calibration/fit parameters differ from registered state')
            scoring_batch_size = detector.score_batch_size
        detector.load_state_dict(detector_payload["state_dict"])
        if detector_name == 'mahalanobis_multilayer_val_calibrated':
            detector.score_batch_size = scoring_batch_size
            if detector.preprocessing_magnitude:
                # Virtual defense-side Lee gradient scores use the same
                # directional/immutable/padded masks as authorized packet
                # edits. This callback is never exposed to attack generation.
                from workflow.calibrate_detector import _editable_mask
                immutable_for_detector = tuple(cfg['dataset'].get('immutable_features', ('DIR','FLG')))
                detector.scoring_mask_builder = lambda data, valid: _editable_mask(
                    np.asarray(data), target_norm, feature_names, np.asarray(valid),
                    immutable_for_detector)
            fitted_params = {'covariance_ridge': detector.covariance_ridge,
                             'preprocessing_magnitude': detector.preprocessing_magnitude,
                             'layers': tuple(detector.layers_)}
            scoring_params = {'score_batch_size': detector.score_batch_size}
        elif detector_name == "mahalanobis":
            detector.score_batch_size = scoring_batch_size
            fitted_params = {"covariance_ridge": detector.covariance_ridge}
            scoring_params = {"score_batch_size": detector.score_batch_size}
        else:
            fitted_params = dict(detector_payload.get("params") or {})
            scoring_params = {}
        detector_artifact = {
            "name": detector_name,
            "checkpoint_sha256": _file_sha256(detector_path),
            "fit_split": "train",
            "fitted_params": fitted_params,
            "scoring_params": scoring_params,
            "target_model_state_sha256": target_digest,
        }
        if logger:
            logger.info("Loaded %s detector from %s; fitted=%s scoring=%s",
                        detector_name, detector_path, fitted_params, scoring_params)

    # -------------------- optional surrogate --------------------
    if progress is not None:
        progress('load_surrogate_and_attack')
    surrogate_wrapper: ModelWrapper | None = None
    attack_input: Any = x_target
    attack_norm = target_norm
    surrogate_meta: dict[str, Any] | None = None
    attack_input_is_target_normalized = False
    surrogate_cfg_raw = cfg.get("surrogate_model")
    surrogate_cfg = dict(surrogate_cfg_raw) if surrogate_cfg_raw else None
    surrogate_enabled = bool(surrogate_cfg.get("enabled", True)) if surrogate_cfg is not None else False
    if surrogate_cfg is not None and surrogate_enabled:
        surrogate_ckpt = store.model_path(surrogate_cfg['name'], 'none', surrogate_seed, role='surrogates')
        store.validate_template(surrogate_cfg.get('checkpoint'), surrogate_ckpt, surrogate_seed, role='surrogate_model')
        store.require('model', surrogate_ckpt, seed=surrogate_seed)
        if surrogate_ckpt.resolve() == target_ckpt.resolve():
            raise ConfigurationError('Surrogate and evaluation target resolve to the same checkpoint path')
        surrogate_meta = _load_checkpoint_metadata(surrogate_ckpt)
        if surrogate_meta.get('experiment_id') != store.read()['experiment_id']:
            raise ConfigurationError('Surrogate checkpoint belongs to another experiment version')
        if budget_profile == 'research' and surrogate_meta.get('training_budget_profile') != 'research':
            raise ConfigurationError('Research transfer evaluation requires a research-trained surrogate checkpoint')
        if (surrogate_meta.get('model_state_sha256') is None or
                surrogate_meta.get('model_state_sha256') == target_meta.get('model_state_sha256')):
            raise ConfigurationError('Authorized surrogate must be independently trained, not an exact copy of the target')
        if surrogate_meta.get("class_mapping", {}) != target_meta.get("class_mapping", {}):
            raise ConfigurationError("Transfer surrogate class-ID mapping differs from target")
        if int(surrogate_meta.get("num_classes", -1)) != int(target_meta["num_classes"]):
            raise ConfigurationError("Transfer surrogate class count differs from target")
        # Check the *same* held-out membership, not just matching seed integers.
        # A flattened surrogate has a different layout identity/hash but must
        # otherwise match source, physical policy, labels, and split IDs.
        surrogate_manifest_path = surrogate_ckpt.parent / "split_manifest.json"
        if not surrogate_manifest_path.is_file():
            raise ConfigurationError("Surrogate split manifest missing; retrain surrogate")
        import json
        surrogate_manifest = json.loads(surrogate_manifest_path.read_text(encoding="utf-8"))
        if manifest_hash(surrogate_manifest) != surrogate_meta.get("split_manifest_sha256"):
            raise ConfigurationError("Surrogate checkpoint/split provenance mismatch")
        for partition in ("train_ids", "validation_ids", "test_ids"):
            if surrogate_manifest.get(partition) != manifest.get(partition):
                raise ConfigurationError("Transfer surrogate training/test membership differs from target")
        # Compare a shallowly different tensor *layout* without mutating
        # the loaded signed manifest's nested transformation dictionaries.
        a = dict(surrogate_manifest.get("identity", {}))
        b = dict(manifest["identity"])
        for identity in (a, b):
            transform = identity.get("transformation")
            if isinstance(transform, dict):
                identity["transformation"] = {k: v for k, v in transform.items() if k != "flatten"}
        if a != b or surrogate_meta.get("dataset_identity") != surrogate_manifest.get("identity"):
            raise ConfigurationError("Transfer surrogate physical transformation/source differs from target")
        surrogate_norm = _normalization_from_checkpoint(
            surrogate_meta,
            feature_names,
            role="surrogate model",
        )
        # Surrogates may use an equivalent (T,F) layout for a flattened
        # target. A validated shape-only adapter runs in bounded batches.
        surrogate_input_shape = tuple(int(i) for i in surrogate_meta['input_shape'])
        if int(np.prod(surrogate_input_shape)) != int(np.prod(x_target.shape[1:])):
            raise ConfigurationError('Target/surrogate coordinate widths differ')
        # Convert target-normalized feature batches to surrogate normalization
        # inside EvaluationEngine rather than allocating x_surrogate for all N.
        surrogate_wrapper = _build_model(
            surrogate_cfg,
            (len(x_target), *surrogate_input_shape),
            device=device_info.torch_device,
            num_classes=int(surrogate_meta["num_classes"]),
            seed=seed,
        )
        _load_checkpoint(surrogate_wrapper, surrogate_ckpt, checkpoint_data=surrogate_meta)
        surrogate_digest = surrogate_meta.get("model_state_sha256")
        if not isinstance(surrogate_digest, str) or frozen_model_state_sha256(surrogate_wrapper) != surrogate_digest:
            raise ConfigurationError("Authorized surrogate checkpoint fingerprint mismatch; retrain surrogate")
        attack_input = x_target
        attack_norm = surrogate_norm
        attack_input_is_target_normalized = True
        if logger:
            logger.info(
                "evaluate.surrogate_loaded seed=%d model=%s checkpoint=%s normalization=on_demand",
                seed, surrogate_cfg["name"], surrogate_ckpt,
            )

    benign_label = int(evaluation_cfg.get("benign_label", 0))

    # The attack receives only the surrogate (if present), never the target model.
    # Every evaluation attack is explicitly directed toward the benign class.
    attack_cls, attack_params = _instantiate_attack(
        cfg,
        feature_names,
        target_label=benign_label,
    )
    attack: BaseAttack = attack_cls(**attack_params)
    access_mode = str(evaluation_cfg.get('attack_access', 'transfer')).lower()
    if access_mode not in {'transfer', 'model_independent'}:
        raise ConfigurationError(
            'Only transfer and model_independent generation are allowed by '
            'the target-isolation contract; target-model white-box access is prohibited'
        )
    if access_mode == 'transfer':
        if surrogate_wrapper is None:
            raise ConfigurationError('Transfer attack requires an explicitly configured surrogate_model')
        attack_model = surrogate_wrapper
    else:
        if surrogate_wrapper is not None:
            raise ConfigurationError('Model-independent attack must not bind a surrogate')
        if getattr(attack_cls, 'requires_gradients', False) or (issubclass(attack_cls, FeatureSpaceAttack) and getattr(attack_cls, 'query_based', False)):
            raise ConfigurationError('Query- and gradient-based attacks require an independently trained surrogate')
        attack_model = None
    # Offline generators must verify exact permitted training membership,
    # physical/source topology, classifier class schema (where applicable),
    # algorithm configuration and original training seed BEFORE generating.
    # Never pass the frozen target's model state to an attack. The surrogate's
    # checkpoint is already verified independently above.
    if str(cfg['attack']['name']) in {'amd_gan', 'transformer_gan', 'nids_gan', 'tdtce', 'a2pm', 'mirage', 'deepred'}:
        source_role = 'surrogate_model' if surrogate_meta is not None else 'model'
        source_manifest = surrogate_manifest if surrogate_meta is not None else manifest
        source_digest = surrogate_digest if surrogate_meta is not None else ''
        source_checkpoint_digest = _file_sha256(surrogate_ckpt) if surrogate_meta is not None else None
        source_shape = tuple(
            int(x) for x in (
                surrogate_meta['input_shape'] if surrogate_meta is not None
                else target_meta['input_shape']
            )
        )
        expected_provenance = generator_artifact_provenance(
            manifest=source_manifest,
            source_state_sha256=source_digest,
            source_checkpoint_sha256=source_checkpoint_digest,
            source_role=source_role,
            attack_config=cfg['attack'],
            seed=attack_seed,
            input_shape=source_shape,
            feature_names=feature_names,
            class_mapping=target_meta['class_mapping'],
            access_mode=access_mode,
        )
        setattr(attack, 'expected_provenance', expected_provenance)
        generator_path = store.generator_path(str(cfg['attack']['name']), str(cfg['attack'].get('variant', 'default')), attack_seed)
        configured_generator = cfg['attack'].get('checkpoint', cfg['attack'].get('params', {}).get('checkpoint'))
        store.validate_template(configured_generator, generator_path, attack_seed, role='attack')
        store.require('generator', generator_path, seed=attack_seed, configuration=cfg['attack'])
    if getattr(attack_cls, "requires_gradients", False) and (
        attack_model is None or not attack_model.capabilities.differentiable
    ):
        raise ConfigurationError(
            f"Attack {attack_cls.__name__} requires input gradients, but the selected attack model "
            f"is {attack_model.capabilities.model_family!r}. Configure a differentiable PyTorch "
            "surrogate_model for attacks against a Random Forest target."
        )
    if surrogate_wrapper is not None and getattr(attack_cls, "requires_gradients", False) and not isinstance(surrogate_wrapper, PyTorchModelWrapper):
        raise ConfigurationError(
            "Gradient-based attacks require a PyTorch surrogate_model; the configured surrogate is not differentiable."
        )
    if logger:
        logger.info(
            "evaluate.attack_ready seed=%d attack=%s params=%s model=%s problem_space=%s",
            seed, attack_cls.__name__, {key: value for key, value in attack_params.items() if key != "feature_names"},
            type(attack_model).__name__, issubclass(attack_cls, ProblemSpaceAttack),
        )

    preprocessor = None
    if issubclass(attack_cls, ProblemSpaceAttack):
        raw_traffic = getattr(dataset, "raw_traffic", None)
        if raw_traffic is None:
            raise ConfigurationError(
                "Problem-space evaluation requires dataset.raw_traffic (PCAP/packet objects) "
                "to be provided by the dataset plugin"
            )
        preprocessing_cfg = dict(cfg.get("preprocessing", {}))
        processor_name = str(preprocessing_cfg.get("name", ""))
        if not processor_name:
            raise ConfigurationError("preprocessing.name is required for problem-space evaluation")
        preprocessor = PREPROCESSORS.get(processor_name)(**dict(preprocessing_cfg.get("params", {})))
        if hasattr(preprocessor, "load_state_dict"):
            preprocessor.load_state_dict(
                {
                    "normalize": True,
                    "mean": target_norm.mean.tolist(),
                    "std": target_norm.std.tolist(),
                }
            )
    else:
        raw_traffic = None

    strict_constraints = bool(evaluation_cfg.get("strict_constraints", True))
    if not strict_constraints and logger is not None:
        logger.warning(
            "NON-STRICT DIAGNOSTIC: physically rejected adversarial candidates "
            "may be scored; this execution cannot certify SOTA feasibility."
        )
    strict_multiclass = bool(evaluation_cfg.get("strict_multiclass", True))
    class_names = tuple(
        str(name) for name in getattr(
            dataset, "classes", tuple(f"Class_{i}" for i in range(target_num_classes))
        )
    )
    gatekeeper = UniversalConstraintGatekeeper(
        strict=strict_constraints,
        normalization_mean=target_norm.mean,
        normalization_std=target_norm.std,
        padding_values={**getattr(dataset, "pad_values", {}), **(
            {str(getattr(dataset, "padding_indicator_name")): 1.0}
            if getattr(dataset, "include_padding_indicator", False) else {}
        )} if hasattr(dataset, "pad_values") else None,
        padding_indicator_name=(str(dataset.padding_indicator_name)
                                if getattr(dataset, "include_padding_indicator", False) else None),
    )
    engine = EvaluationEngine(
        target_wrapper,
        attack,
        gatekeeper,
        attack_model=attack_model,
        surrogate_input_shape=tuple(surrogate_meta["input_shape"]) if access_mode == "transfer" else None,
        preprocessor=preprocessor,
        attack_normalization=attack_norm,
        target_normalization=target_norm,
        class_labels=list(range(target_num_classes)),
        class_names=class_names,
        benign_label=benign_label,
        target_label=attack_params.get("target_label"),
        c_fp=float(evaluation_cfg.get("c_fp", 1.0)),
        c_fn=float(evaluation_cfg.get("c_fn", 1.0)),
        strict_constraints=strict_constraints,
        strict_multiclass=strict_multiclass,
        detector=detector,
        batch_size=int(evaluation_cfg.get("batch_size", 256)),
        max_samples=max_samples,
        logger=logger,
    )

    valid_mask = getattr(dataset, "valid_mask", None)
    if sample_indices is not None:
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask)[sample_indices]
        if raw_traffic is not None:
            if isinstance(raw_traffic, np.ndarray):
                raw_traffic = raw_traffic[sample_indices]
            else:
                raw_traffic = [raw_traffic[index] for index in sample_indices]
    
    # Keep features and labels as NumPy arrays on the CPU.
    # EvaluationEngine will slice them, and the individual Attack strategies will move the slices to the GPU dynamically per-batch.
    y_attack = y_np
    attack_features = (
        attack_input
        if not issubclass(attack_cls, ProblemSpaceAttack) 
        else None
    )

    declared_budget = (
        attack.final_budget_contract() if isinstance(attack, FeatureSpaceAttack) else None
    )
    if progress is not None:
        progress('evaluate_clean_and_attacked')
    output = engine.run(
        x_target,
        y_attack,
        feature_names=feature_names,
        attack_input=attack_features,
        attack_input_is_target_normalized=(
            attack_input_is_target_normalized and not issubclass(attack_cls, ProblemSpaceAttack)
        ),
        clean_physical=clean_physical,
        raw_traffic=raw_traffic,
        valid_mask=valid_mask,
        attack_kwargs={"seed": int(attack_seed)},
        progress=progress,
    )

    if progress is not None:
        progress('serialize_evaluation_results')
    clean_metrics = classification_metrics(
        y_np,
        output.baseline_predictions,
        labels=np.arange(target_num_classes, dtype=np.int64),
    ).as_dict()
    metrics = dict(output.metrics)
    output_run = Path(cfg.get("output_dir", "../runs")) / "per_seed"
    output_run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_run / f"matched_outcomes_seed{seed}.npz",
        sample_ids=np.asarray(dataset.sample_ids, dtype=str)[sample_indices],
        true_labels=y_np,
        clean_predictions=output.baseline_predictions,
        adversarial_predictions_scored_only=output.adversarial_predictions,
        constraint_gated_defended_predictions=output.defended_predictions,
        accepted_mask=output.accepted_mask,
        scored_mask=output.scored_mask,
        group_ids=(np.asarray(dataset.group_ids, dtype=str)[sample_indices]
                   if getattr(dataset, "group_ids", None) is not None else np.asarray([], dtype=str)),
        # Per-record counts are a plugin-attributed subset of the independent
        # query ledger. -1 means no reliable per-record attribution exists.
        surrogate_inference_queries_per_sample=(
            np.asarray(output.attack_result.queries, dtype=np.int64)
            if output.attack_result.queries is not None else
            np.full(len(y_np), -1, dtype=np.int64)
        ),
    )
    from core.provenance import _digest_strings
    import hashlib
    import json
    selected_ids = [str(dataset.sample_ids[i]) for i in sample_indices]
    # A SHA256 of the exact held-out ordered IDs is required for pairing.
    # Individual IDs are retained in the NPZ for per-record paired analyses.
    selected_test_ids_digest = _digest_strings(selected_ids)
    for key, value in clean_metrics.items():
        metrics[f"baseline_{key}"] = float(value)
    # A structured support record covers every population, including the
    # accepted subset when no candidate survived strict physical admission.
    # The metrics dictionary retains flat numeric support fields for aggregators.
    class_labels = tuple(range(target_num_classes))
    class_support = {
        'full_test_clean': classification_support(y_np, output.baseline_predictions, class_labels),
        'full_test_constraint_gated_defended': classification_support(
            y_np, output.defended_predictions, class_labels,
        ),
        'accepted_subset_clean': classification_support(
            y_np[output.accepted_mask], output.baseline_predictions[output.accepted_mask], class_labels,
        ),
        'accepted_subset_adversarial': classification_support(
            y_np[output.accepted_mask], output.adversarial_predictions[output.accepted_mask], class_labels,
        ),
    }
    metrics["attack_generated_sample_count"] = float(len(y_np))
    metrics["surrogate_used"] = float(surrogate_wrapper is not None)
    metrics["benchmark_protocol_version"] = 2.0
    metrics["test_partition_samples"] = float(len(manifest["test_ids"]))
    metrics["dataset_truncated_flows_total"] = float(getattr(dataset, "truncated_flow_count", 0))
    metrics["dataset_discarded_packet_positions_total"] = float(getattr(dataset, "discarded_packet_positions", 0))
    metrics["dataset_truncated_flows_rate"] = (
        metrics["dataset_truncated_flows_total"] / len(dataset) if len(dataset) else float("nan")
    )
    if output.detector_clean_scores is not None and output.detector_adversarial_scores is not None:
        detector_scores_dir = Path(cfg.get("output_dir", "../runs")) / "per_seed" / "detector_scores"
        detector_scores_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            detector_scores_dir / f"detector_scores_seed{seed}.npz",
            clean_scores=output.detector_clean_scores,
            adversarial_scores=output.detector_adversarial_scores,
            accepted_mask=output.accepted_mask,
            labels=y_np,
        )
    if surrogate_meta is not None:
        metrics["surrogate_num_classes"] = float(surrogate_meta["num_classes"])

    if ExperimentReporter is not None:
        reporter = ExperimentReporter(
            run_dir=Path(cfg.get("output_dir", "../runs")),
            class_names=class_names,
        )
        y_true_scored = y_np[output.scored_mask]
        y_pred_scored = output.adversarial_predictions[output.scored_mask]
        reporter.save_predictions(
            seed, y_true_scored, output.baseline_predictions[output.scored_mask], y_pred_scored
        )
        reporter.save_seed_report(seed, metrics, y_true_scored, y_pred_scored)
        reporter.generate_confusion_matrix(seed, y_true_scored, y_pred_scored)

    if logger:
        logger.info(
            "evaluate.report_saved seed=%d asr=%.2f%% realizability=%.2f%% "
            "scored=%d rejected=%d",
            seed, 100.0 * metrics.get("asr", 0.0),
            100.0 * metrics.get("realizability_rate", 0.0),
            int(metrics.get("scored_sample_count", 0.0)),
            int(metrics.get("rejected_sample_count", 0.0)),
        )

    result_dir = Path(cfg.get("output_dir", "../runs")) / "per_seed"
    _save_json(
        result_dir / f"evaluate_seed_{seed}.json",
        {
            "seed": int(seed),
            "target_model": target_cfg.get("name"),
            "surrogate_model": None if surrogate_cfg is None else surrogate_cfg.get("name"),
            "protocol": {
                "source_revision_sha256": source_revision_sha256(),
                "experiment_id": store.read()['experiment_id'],
                "attack_name": str(cfg['attack']['name']),
                "target_defense_mode": str(target_cfg.get('training', {}).get('defense_mode', 'none')),
                "target_architecture_params_sha256": hashlib.sha256(
                    json.dumps(target_meta['model_params'], sort_keys=True, default=str).encode('utf-8')
                ).hexdigest(),
                "seed_roles": {"split": int(manifest["seed"]), "target_model": int(target_seed), "surrogate": int(surrogate_seed) if surrogate_meta else None, "attack": int(attack_seed), "detector": int(detector_seed) if detector else None, "worker_replica": int(seed)},
                "budget_profile": budget_profile,
                "classification_metric_policy_version": CLASSIFICATION_METRIC_POLICY_VERSION,
                "split_policy_version": manifest.get('split_policy', {}).get('version'),
                "comparable_budget_id": evaluation_cfg.get("comparable_budget_id"),
                "threat_model_id": "mapping_v1_problem_space" if issubclass(attack_cls, ProblemSpaceAttack) else "feature_space",
                "class_mapping_sha256": hashlib.sha256(json.dumps(target_meta["class_mapping"],sort_keys=True).encode()).hexdigest(),
                "physical_transformation_sha256": hashlib.sha256(json.dumps(manifest["identity"].get("transformation"),sort_keys=True).encode()).hexdigest(),
                "target_checkpoint_sha256": __import__("core.provenance", fromlist=["_file_sha256"])._file_sha256(target_ckpt),
                "detector_artifact": detector_artifact,
                "surrogate_checkpoint_sha256": (
                    __import__("core.provenance", fromlist=["_file_sha256"])._file_sha256(surrogate_ckpt)
                    if surrogate_meta is not None else None
                ),
                "manifest_sha256": manifest_hash(manifest),
                "selected_test_ids_sha256": selected_test_ids_digest,
                "attack_access": access_mode,
                "final_hard_budget": (
                    {"norm": declared_budget.norm,
                     "epsilon": declared_budget.epsilon,
                     "domain": declared_budget.domain,
                     "normalization_role": "authorized_surrogate_or_attack",
                     "normalization_sha256": hashlib.sha256(
                         np.ascontiguousarray(attack_norm.mean, dtype=np.float32).tobytes()
                         + np.ascontiguousarray(attack_norm.std, dtype=np.float32).tobytes()
                     ).hexdigest()}
                    if declared_budget is not None else None
                ),
                "final_budget_policy_version": "post_projection_hard_budget_v1",
                "padding_occupancy_policy_version": "canonical_prefix_v1",
                "query_accounting_policy_version": "observed_oracle_operations_v1",
                "strict_constraints": strict_constraints,
                "evaluation_status": (
                    "strict_feasible_benchmark" if strict_constraints
                    else "non_strict_diagnostic_not_sota_certified"
                ),
                "dataset_source_sha256": manifest["identity"]["source_sha256"],
                "target_label": int(benign_label),
                "sequence_policy": manifest["identity"].get("transformation", {}).get("sequence_policy") if manifest["identity"].get("transformation") else None,
            },
            "metrics": metrics,
            "class_support_by_population": class_support,
        },
    )
    return metrics
