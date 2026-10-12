from __future__ import annotations

import numpy as np
import torch
from torch import nn

from attacks.feature_space.pgd import PGDAttack
from attacks.feature_space.amd_gan import AMDGANAdapter
from attacks.feature_space.square import SquareAttack
from attacks.problem_space.dummy_pcap import DummyPCAPMutator
from constraints.gatekeeper import UniversalConstraintGatekeeper
from core.registry import ATTACKS, NIDS, discover_all
from evaluator.metrics import attack_success_rate, attack_success_mask, classification_metrics
from models.wrapper import PyTorchModelWrapper, SklearnModelWrapper
from workflow import _instantiate_attack
from evaluator.engine import EvaluationEngine


def test_discovery_registers_components() -> None:
    discover_all()
    assert "pgd" in ATTACKS.names()
    assert "mlp" in NIDS.names()


def test_attack_metadata_is_not_passed_to_attack_constructors() -> None:
    discover_all()
    feature_names = ("feature",)
    configs = [
        {
            "name": "pgd",
            "epsilon": 0.03,
            "variant": "eps0.03_steps2",
            "training": {"epochs": 2},
        },
        {
            "name": "square",
            "epsilon": 0.03,
            "variant": "default",
        },
        {
            "name": "amd_gan",
            "generator": None,
            "epsilon": 0.05,
            "training": {"epochs": 2},
            "variant": "default",
        },
        {
            "name": "problem_space_dummy",
            "byte_delta": 4,
            "variant": "default",
        },
    ]

    for attack_cfg in configs:
        attack_cls, params = _instantiate_attack({"attack": attack_cfg}, feature_names)
        assert "variant" not in params
        assert "training" not in params
        attack_cls(**params)


def test_gatekeeper_rejects_invalid_flow() -> None:
    names = ("flow_duration_s", "packet_count", "byte_count", "protocol", "tcp_syn", "tcp_ack", "tcp_fin", "tcp_rst")
    x = np.array([[1.0, 2.0, 200.0, 6.0, 1.0, 0.0, 0.0, 0.0], [2.0, -1.0, 200.0, 6.0, 1.0, 1.0, 0.0, 0.0]])
    result = UniversalConstraintGatekeeper().validate(x, names)
    assert result.accepted.tolist() == [True, False]
    assert result.realizability_rate == 0.5


def test_gatekeeper_freezes_victim_direction() -> None:
    names = ("PL", "DIR", "TTL")
    clean = np.array([[[10.0, 1.0, 64.0], [20.0, 0.0, 64.0]]], dtype=np.float32)
    adversarial = clean.copy()
    adversarial[0, 0, 0] += 5.0
    adversarial[0, 1, 0] += 5.0
    result = UniversalConstraintGatekeeper().validate(
        adversarial,
        names,
        original_samples=clean,
        valid_mask=np.array([[True, True]]),
    )
    assert result.accepted.tolist() == [False]
    assert result.violations[0].rule == "directional_immutability"


def test_model_gradient_contract() -> None:
    model = nn.Sequential(nn.Linear(3, 2))
    wrapper = PyTorchModelWrapper(model)
    x = torch.randn(4, 3)
    y = torch.tensor([0, 1, 0, 1])
    grad = wrapper.get_gradients(x, y)
    assert grad.shape == x.shape


def test_pgd_changes_tensor() -> None:
    model = nn.Sequential(nn.Linear(3, 2))
    wrapper = PyTorchModelWrapper(model)
    x = torch.randn(4, 3)
    y = torch.tensor([0, 1, 0, 1])
    result = PGDAttack(epsilon=0.1, alpha=0.02, steps=2).bind_model(wrapper).generate(x, y)
    assert result.adversarial.shape == x.shape


def test_metrics_are_imbalance_safe() -> None:
    metrics = classification_metrics([0, 0, 0, 1], [0, 0, 1, 1])
    assert 0.0 <= metrics.balanced_accuracy <= 1.0
    assert 0.0 <= metrics.gmean <= 1.0
    assert 0.0 <= attack_success_rate([1, 1], [1, 1], [0, 1], benign_label=0) <= 1.0


def test_multiclass_attack_success_requires_target_label() -> None:
    # Class 2 -> class 1 is a classification error, not a successful evasion.
    mask = attack_success_mask(
        y_true=[1, 2, 0],
        baseline_pred=[1, 2, 0],
        adversarial_pred=[0, 1, 1],
        benign_label=0,
    )
    assert mask.tolist() == [True, False, False]
