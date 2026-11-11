"""Multi-layer Lee-style Gaussian detector (Lee et al., NeurIPS 2018, Sec. 2).

Reproduces per-layer class-conditional tied-covariance Mahalanobis scores and
validation-calibrated logistic combination using actual PyTorch hidden features.
Mandatory NIDS adaptation: masked real-packet pooling and training-only class
Gaussian fitting; the frozen target is accessed only in defender preparation or
post-generation evaluation, never in attack generation. Calibration requires
separately authorized held-out *validation* clean/attack scores, not test data.
Original Algorithm 1 per-layer confidence-gradient preprocessing is available
only as optional masked *detector scoring*, never as a physical attack mutation.
Not yet registered: authorized adversarial-validation generation, target/split
registry orchestration and evaluation-time cost attribution are outstanding.
No deployed traffic-blocking threshold or original-equivalence claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from core.registry import register_detector
from core.provenance import frozen_model_state_sha256, _digest_strings
from models.wrapper import PyTorchModelWrapper
from .base import BaseDetector


@register_detector('mahalanobis_multilayer_val_calibrated')
class LeeMultilayerMahalanobis(BaseDetector):
    """Layerwise nearest-class Mahalanobis scores, combined on validation only.

    This is an opt-in detector for *actual neural* intermediate layers. Scoring
    cannot occur until its separate authorized validation calibration finishes.
    Higher calibrated probabilities mean greater deviation from clean traffic.
    """

    REGISTRY_NAME = 'mahalanobis_multilayer_val_calibrated'
    STATE_VERSION = 2

    def __init__(self, covariance_ridge: float = 1e-5, score_batch_size: int = 1024,
                 preprocessing_magnitude: float = 0.0) -> None:
        if covariance_ridge <= 0 or not np.isfinite(covariance_ridge):
            raise ValueError('covariance_ridge must be a positive finite value')
        if int(score_batch_size) < 1:
            raise ValueError('score_batch_size must be positive')
        if not np.isfinite(preprocessing_magnitude) or preprocessing_magnitude < 0:
            raise ValueError('preprocessing_magnitude must be finite and nonnegative')
        self.covariance_ridge = float(covariance_ridge)
        self.score_batch_size = int(score_batch_size)
        self.preprocessing_magnitude = float(preprocessing_magnitude)
        # Runtime edit-mask function is supplied only after the fitted target,
        # normalization and immutable schema are registered and verified.
        # It is never part of the fitted checkpoint or an attack-side oracle.
        self.scoring_mask_builder = None
        self.last_detector_gradient_forward_records = 0
        self.last_detector_inference_records = 0
        self.layers_: list[str] = []
        self.class_labels_: np.ndarray | None = None
        self.means_: dict[str, np.ndarray] = {}
        self.precisions_: dict[str, np.ndarray] = {}
        self.n_fit_samples_: int = 0
        self.calibration_coef_: np.ndarray | None = None
        self.calibration_intercept_: float | None = None
        self.calibration_membership_sha256: str | None = None
        self.target_model_state_sha256: str | None = None
        self.split_manifest_sha256: str | None = None
        self.train_membership_sha256: str | None = None
        self._registered_train_ids: tuple[str,...] = ()
        self.calibration_attack_source: str | None = None

    @staticmethod
    def _validate_wrapper(wrapper: Any) -> PyTorchModelWrapper:
        if not isinstance(wrapper, PyTorchModelWrapper):
            raise TypeError('Lee multi-layer detector requires PyTorchModelWrapper with actual hidden states')
        return wrapper

    def _extract(self, x: Any, wrapper: PyTorchModelWrapper, valid_mask: Any = None) -> dict[str, np.ndarray]:
        result = wrapper.extract_intermediate_features(x, valid_mask=valid_mask)
        arrays = {name: value.detach().to(dtype=torch.float64).cpu().numpy()
                  for name, value in result.items()}
        if any(v.ndim != 2 or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Layer features must be finite and batch-first')
        return arrays

    def _batch(self, batch: Any, wrapper: PyTorchModelWrapper) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError('Detector batches must contain input and label')
        x, y = batch[:2]
        feat = self._extract(x, wrapper, batch[2] if len(batch) > 2 else None)
        labels = np.asarray(y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y).reshape(-1)
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('Gaussian fitting requires integer class labels')
        if any(len(v) != len(labels) for v in feat.values()):
            raise ValueError('Mismatched label and feature batch')
        return feat, labels.astype(np.int64, copy=False)

    def fit(self, dataloader: Any, wrapper: PyTorchModelWrapper, *,
            registered_train_ids: Sequence[str] | None = None,
            registered_split_manifest_sha256: str | None = None) -> 'LeeMultilayerMahalanobis':
        """Fit per-layer class means and tied covariance on clean TRAIN only.

        Two bounded passes avoid keeping the full corpus or physical float64
        packet arrays in memory. A reiterable, deterministic loader is required.
        Fitting alone does not make the detector ready for scoring: independent
        validation calibration must be supplied and completed separately.
        """
        wrapper = self._validate_wrapper(wrapper)
        if (registered_train_ids is None) != (registered_split_manifest_sha256 is None):
            raise ValueError('Registered training IDs and split digest must be supplied together')
        if registered_split_manifest_sha256 is not None and not self._sha_valid(registered_split_manifest_sha256):
            raise ValueError('Invalid registered split manifest digest')
        expected_ids = None if registered_train_ids is None else tuple(map(str, registered_train_ids))
        if expected_ids is not None and (not expected_ids or len(expected_ids) != len(set(expected_ids))):
            raise ValueError('Registered training IDs must be nonempty and unique')
        observed_ids: list[str] = []
        sums: dict[str, dict[int, np.ndarray]] = {}
        counts: dict[int, int] = {}
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                if expected_ids is not None:
                    observed_ids.extend(self._batch_ids(batch, 'train'))
                feat, y = self._batch(batch, wrapper)
                if not feat or not len(y):
                    continue
                if not sums:
                    sums = {name: {} for name in feat}
                elif tuple(feat) != tuple(sums):
                    raise ValueError('Hidden layer order changed during fit')
                for klass in np.unique(y):
                    key = int(klass)
                    chosen = y == klass
                    counts[key] = counts.get(key, 0) + int(chosen.sum())
                    for name, data in feat.items():
                        layer_sums = sums[name]
                        part = data[chosen].sum(axis=0, dtype=np.float64)
                        if key not in layer_sums:
                            layer_sums[key] = part
                        else:
                            layer_sums[key] += part
                n += len(y)
            if n == 0 or len(counts) < 2:
                raise ValueError('Need labeled clean training examples from at least two classes')
            if expected_ids is not None and tuple(observed_ids) != expected_ids:
                raise ValueError('Observed clean training IDs differ from registered train partition')
            classes = np.array(sorted(counts), dtype=np.int64)
            means = {name: np.stack([sums[name][int(c)] / counts[int(c)] for c in classes])
                     for name in sums}
            covariance = {name: np.zeros((mu.shape[1], mu.shape[1]), dtype=np.float64)
                          for name, mu in means.items()}
            second_counts = {int(c): 0 for c in classes}
            second_ids: list[str] = []
            for batch in dataloader:
                if expected_ids is not None:
                    second_ids.extend(self._batch_ids(batch, 'train'))
                feat, y = self._batch(batch, wrapper)
                if tuple(feat) != tuple(means):
                    raise ValueError('Hidden layer order changed between fit passes')
                for c in np.unique(y):
                    key = int(c)
                    if key not in second_counts:
                        raise ValueError('Unexpected class in covariance pass')
                    selected = y == c
                    second_counts[key] += int(selected.sum())
                    index = int(np.searchsorted(classes, key))
                    for name, features in feat.items():
                        residual = features[selected] - means[name][index]
                        covariance[name] += residual.T @ residual
        if second_counts != counts or (expected_ids is not None and tuple(second_ids) != expected_ids):
            raise ValueError('Training membership changed between Gaussian passes')
        precisions: dict[str, np.ndarray] = {}
        for name, matrix in covariance.items():
            matrix /= n
            matrix = 0.5 * (matrix + matrix.T)
            matrix.flat[::len(matrix)+1] += self.covariance_ridge
            p = np.linalg.pinv(matrix, hermitian=True)
            if not np.isfinite(p).all():
                raise ValueError(f'Nonfinite Gaussian precision for layer {name}')
            precisions[name] = p
        self.layers_ = list(means)
        self.class_labels_ = classes
        self.means_ = means
        self.precisions_ = precisions
        self.n_fit_samples_ = n
        self.calibration_coef_ = None
        self.calibration_intercept_ = None
        self.calibration_membership_sha256 = None
        self.target_model_state_sha256 = frozen_model_state_sha256(wrapper)
        self._registered_train_ids = () if expected_ids is None else expected_ids
        self.train_membership_sha256 = None if expected_ids is None else _digest_strings(expected_ids)
        self.split_manifest_sha256 = registered_split_manifest_sha256
        self.calibration_attack_source = None
        return self

    def raw_layer_scores(self, x: Any, wrapper: PyTorchModelWrapper, valid_mask: Any = None,
                         *, score_perturbation_mask: Any = None) -> np.ndarray:
        """Return per-layer Gaussian scores, optionally with Lee input preprocessing.

        The small input-gradient step is a *virtual detector computation* in
        normalized float32 space, not a new physical packet or attack proposal.
        It is restricted to externally verified editable coordinates; the
        frozen target is never accessible to attack generation through this API.
        """
        wrapper = self._validate_wrapper(wrapper)
        self._check_target(wrapper)
        if not self.layers_ or self.class_labels_ is None:
            raise RuntimeError('Fit clean training Gaussians before scoring')
        if self.preprocessing_magnitude:
            return self._preprocessed_layer_scores(x, wrapper, valid_mask,
                                                  score_perturbation_mask)
        self.last_detector_gradient_forward_records = 0
        self.last_detector_inference_records = len(x)
        parts = self._extract(x, wrapper, valid_mask)
        if list(parts) != self.layers_:
            raise ValueError('Detector hidden layer identity does not match fitted model')
        columns = []
        for name in self.layers_:
            x_layer, means, precision = parts[name], self.means_[name], self.precisions_[name]
            if x_layer.shape[1] != means.shape[1]:
                raise ValueError('Hidden-layer dimensions changed after fitting')
            delta = x_layer[:, None, :] - means[None, :, :]
            distance = np.einsum('bcd,de,bce->bc', delta, precision, delta)
            columns.append(-distance.min(axis=1))
        return np.stack(columns, axis=1)

    def _preprocessed_layer_scores(self, x: Any, wrapper: PyTorchModelWrapper,
                                   valid_mask: Any, allowed_mask: Any) -> np.ndarray:
        """Lee Algorithm 1 sign-gradient confidence ascent, layer by layer.

        Input is a detector-only normalized tensor. No physical reconstruction,
        attack score, or target-query permission is transferred to a generator.
        Caller must supply the mask which has already excluded immutable,
        victim-owned, and padded coordinates under the registered feature schema.
        """
        source = wrapper._tensor(x).detach()
        if allowed_mask is None:
            raise ValueError('Lee gradient preprocessing requires an externally validated edit mask')
        permitted = torch.as_tensor(allowed_mask, device=source.device, dtype=torch.bool)
        if permitted.shape != source.shape:
            raise ValueError('Lee edit mask shape must exactly match detector input')
        if source.ndim == 3:
            valid = torch.as_tensor(valid_mask,device=source.device,dtype=torch.bool)
            if valid.shape != source.shape[:2] or bool((permitted & ~valid[:,:,None]).any()):
                raise ValueError('Lee detector preprocessing cannot modify padded packet features')
        output: list[np.ndarray] = []
        for name in self.layers_:
            variable = source.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                hidden = wrapper.extract_intermediate_features(
                    variable, valid_mask=valid_mask, differentiable=True)
                layer = hidden[name]
                means = torch.as_tensor(self.means_[name], dtype=layer.dtype, device=layer.device)
                precision = torch.as_tensor(self.precisions_[name], dtype=layer.dtype, device=layer.device)
                residual = layer[:,None,:] - means[None,:,:]
                dist = torch.einsum('bcd,de,bce->bc', residual, precision, residual)
                confidence = -dist.min(1).values
                gradient = torch.autograd.grad(confidence.sum(), variable)[0]
            if not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite Lee detector input-confidence gradient')
            virtual = source + self.preprocessing_magnitude * gradient.sign().detach() * permitted
            hidden_pre = wrapper.extract_intermediate_features(virtual.detach(), valid_mask=valid_mask)
            feat = hidden_pre[name].to(dtype=torch.float64).cpu().numpy()
            difference = feat[:,None,:] - self.means_[name][None,:,:]
            d = np.einsum('bcd,de,bce->bc', difference, self.precisions_[name], difference)
            output.append(-d.min(axis=1))
        self.last_detector_gradient_forward_records = len(source) * len(self.layers_)
        self.last_detector_inference_records = len(source) * len(self.layers_)
        return np.stack(output,axis=1)

    @staticmethod
    def _sha_valid(value: Any) -> bool:
        return (isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value))

    @staticmethod
    def _batch_ids(batch: Any, partition: str) -> tuple[str,...]:
        # The fourth loader component must be the actual observed stable IDs,
        # not a digest asserted by the caller. The third remains valid_mask.
        if not isinstance(batch, (list, tuple)) or len(batch) < 4:
            raise ValueError(f'{partition} loader must expose observed sample IDs as component four')
        ids = batch[3]
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return tuple(map(str, ids))

    def _check_target(self, wrapper: PyTorchModelWrapper) -> None:
        if (self.target_model_state_sha256 is None
                or frozen_model_state_sha256(wrapper) != self.target_model_state_sha256):
            raise ValueError('Detector target model state fingerprint differs from fitted checkpoint')

    def calibrate(self, clean_validation_loader: Any, adversarial_validation_loader: Any,
                  wrapper: PyTorchModelWrapper, *, partition: str,
                  membership_sha256: str, registered_validation_ids: Sequence[str] | None = None,
                  registered_split_manifest_sha256: str | None = None,
                  attack_source: str | None = None) -> 'LeeMultilayerMahalanobis':
        """Calibrate on independent validation-only clean/attacked score pairs.

        It is explicitly forbidden to use test membership, or to estimate
        adversarial validation membership by evaluating the frozen target from
        the attack generator. Caller must provide previously authorized loaders.
        """
        if partition != 'validation':
            raise ValueError('Detector calibration must use validation partition only')
        if not self._sha_valid(membership_sha256):
            raise ValueError('Calibration requires a canonical SHA256 of validation membership')
        if (self.split_manifest_sha256 is None or registered_split_manifest_sha256 != self.split_manifest_sha256
                or not self._registered_train_ids):
            raise ValueError('Calibration requires verified registered train/split membership')
        expected = () if registered_validation_ids is None else tuple(map(str, registered_validation_ids))
        if (not expected or len(expected) != len(set(expected))
                or set(expected).intersection(self._registered_train_ids)
                or _digest_strings(expected) != membership_sha256):
            raise ValueError('Registered validation membership is missing, overlapping train, or mismatched')
        if not isinstance(attack_source, str) or not attack_source.strip():
            raise ValueError('Calibration requires a documented independent-surrogate attack source')
        self._validate_wrapper(wrapper)
        self._check_target(wrapper)
        groups = []
        for loader in (clean_validation_loader, adversarial_validation_loader):
            observed, observed_ids = [], []
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or not len(batch):
                    raise ValueError('Validation loader must provide input batches')
                observed_ids.extend(self._batch_ids(batch, 'validation'))
                observed.append(self.raw_layer_scores(batch[0], wrapper,
                                batch[2] if len(batch) > 2 else None,
                                score_perturbation_mask=batch[4] if len(batch) > 4 else None))
            if tuple(observed_ids) != expected:
                raise ValueError('Observed validation loader IDs do not match registered validation membership')
            if not observed:
                raise ValueError('Both calibration populations must contain examples')
            groups.append(np.concatenate(observed, axis=0))
        x = np.concatenate(groups, axis=0)
        y = np.concatenate([np.zeros(len(groups[0])), np.ones(len(groups[1]))]).astype(int)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite calibration scores')
        model = LogisticRegression(solver='lbfgs', max_iter=500, random_state=0)
        model.fit(x, y)
        self.calibration_coef_ = model.coef_[0].astype(np.float64, copy=True)
        self.calibration_intercept_ = float(model.intercept_[0])
        self.calibration_membership_sha256 = membership_sha256
        self.calibration_attack_source = attack_source
        return self

    def score(self, x: Any, wrapper: PyTorchModelWrapper, valid_mask: Any = None,
              *, score_perturbation_mask: Any = None) -> np.ndarray:
        if self.calibration_coef_ is None or self.calibration_intercept_ is None:
            raise RuntimeError('Detector requires independent validation calibration before score')
        # score_batch_size limits actual detector inference, not just metadata.
        # Work on the caller's view: no additional copy of the whole corpus.
        if self.preprocessing_magnitude and score_perturbation_mask is None:
            if self.scoring_mask_builder is None:
                raise ValueError('Lee detector needs runtime-verified physical edit-mask policy')
            score_perturbation_mask = self.scoring_mask_builder(x, valid_mask)
        n = len(x)
        parts: list[np.ndarray] = []
        gradients = inference = 0
        for start in range(0, n, self.score_batch_size):
            stop = min(start+self.score_batch_size, n)
            chunk_mask = None if valid_mask is None else valid_mask[start:stop]
            chunk_permitted = (None if score_perturbation_mask is None
                               else score_perturbation_mask[start:stop])
            raw = self.raw_layer_scores(x[start:stop], wrapper, chunk_mask,
                                        score_perturbation_mask=chunk_permitted)
            gradients += self.last_detector_gradient_forward_records
            inference += self.last_detector_inference_records
            if raw.shape != (stop-start, len(self.calibration_coef_)):
                raise ValueError('Calibrated layer dimensions do not match')
            z = raw @ self.calibration_coef_ + self.calibration_intercept_
            parts.append(np.exp(-np.logaddexp(0., -z)))
        self.last_detector_gradient_forward_records = gradients
        self.last_detector_inference_records = inference
        return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float64)

    def state_dict(self) -> dict[str, Any]:
        if not self.layers_:
            raise RuntimeError('Cannot serialize an unfitted detector')
        return {
            'version': self.STATE_VERSION, 'covariance_ridge': self.covariance_ridge,
            'score_batch_size': self.score_batch_size,
            'preprocessing_magnitude': self.preprocessing_magnitude,
            'layers': list(self.layers_),
            'class_labels': self.class_labels_.copy(),
            'means': {k: v.copy() for k,v in self.means_.items()},
            'precisions': {k: v.copy() for k,v in self.precisions_.items()},
            'n_fit_samples': self.n_fit_samples_,
            'calibration_coef': None if self.calibration_coef_ is None else self.calibration_coef_.copy(),
            'calibration_intercept': self.calibration_intercept_,
            'calibration_membership_sha256': self.calibration_membership_sha256,
            'target_model_state_sha256': self.target_model_state_sha256,
            'train_membership_sha256': self.train_membership_sha256,
            'split_manifest_sha256': self.split_manifest_sha256,
            'calibration_attack_source': self.calibration_attack_source,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get('version', -1)) != self.STATE_VERSION:
            raise ValueError('Unsupported Lee multilayer detector state version')
        names = list(state['layers'])
        means = {k: np.asarray(state['means'][k], dtype=np.float64).copy() for k in names}
        precisions = {k: np.asarray(state['precisions'][k], dtype=np.float64).copy() for k in names}
        classes = np.asarray(state['class_labels'], dtype=np.int64).copy()
        if len(names) < 2 or len(names) != len(set(names)):
            raise ValueError('Invalid hidden-layer identities in calibration state')
        for name in names:
            mu, p = means[name], precisions[name]
            if mu.ndim != 2 or mu.shape[0] != len(classes) or p.shape != (mu.shape[1], mu.shape[1]):
                raise ValueError('Invalid fitted Gaussian layer shape')
            if not np.isfinite(mu).all() or not np.isfinite(p).all():
                raise ValueError('Nonfinite fitted Gaussian statistics')
        coef = state.get('calibration_coef')
        if coef is not None:
            coef = np.asarray(coef, dtype=np.float64).copy()
            if coef.shape != (len(names),) or not np.isfinite(coef).all():
                raise ValueError('Invalid calibration coefficient dimensions')
            if state.get('calibration_intercept') is None or state.get('calibration_membership_sha256') is None:
                raise ValueError('Incomplete calibrated detector membership')
        if not self._sha_valid(state.get('target_model_state_sha256')):
            raise ValueError('Lee state lacks fitted target checkpoint fingerprint')
        if coef is not None and (not self._sha_valid(state.get('train_membership_sha256'))
                or not self._sha_valid(state.get('split_manifest_sha256'))
                or not state.get('calibration_attack_source')):
            raise ValueError('Calibrated Lee state lacks registered training/validation provenance')
        self.covariance_ridge = float(state['covariance_ridge'])
        self.score_batch_size = int(state['score_batch_size'])
        self.preprocessing_magnitude = float(state['preprocessing_magnitude'])
        if self.preprocessing_magnitude < 0 or not np.isfinite(self.preprocessing_magnitude):
            raise ValueError('Invalid Lee preprocessing magnitude in checkpoint')
        self.layers_ = names
        self.class_labels_ = classes
        self.means_ = means
        self.precisions_ = precisions
        self.n_fit_samples_ = int(state['n_fit_samples'])
        self.calibration_coef_ = coef
        self.calibration_intercept_ = None if state.get('calibration_intercept') is None else float(state['calibration_intercept'])
        self.calibration_membership_sha256 = state.get('calibration_membership_sha256')
        self.target_model_state_sha256 = state['target_model_state_sha256']
        self.train_membership_sha256 = state.get('train_membership_sha256')
        self.split_manifest_sha256 = state.get('split_manifest_sha256')
        self.calibration_attack_source = state.get('calibration_attack_source')
        self._registered_train_ids = ()  # reload uses persisted hashed provenance

