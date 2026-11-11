"""Independent final adversarial-budget verification (not an attack projector).

Physical float64 is the only source of truth. A declared normalized-space
budget is checked by *forward encoding* final and pristine physical values
using the authorized attack/surrogate normalization, NEVER inverse decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from core.dual_representation import encode_normalized32


@dataclass(frozen=True)
class FinalAttackBudget:
    norm: str
    epsilon: float
    domain: str = "normalized_float32"

    def __post_init__(self) -> None:
        if self.domain != "normalized_float32":
            raise ValueError("Unsupported attack budget domain; declare the optimizer's actual domain")
        if self.norm not in {"linf", "l2"}:
            raise ValueError("Unsupported hard attack budget norm")
        if (not np.isfinite(self.epsilon) or self.epsilon < 0
                or self.epsilon > float(np.finfo(np.float32).max)):
            raise ValueError("Attack epsilon must be finite, nonnegative and representable in float32")


def check_final_budget(
    final_physical: np.ndarray,
    clean_physical: np.ndarray,
    feature_names: Sequence[str],
    budget: FinalAttackBudget,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray:
    """Return one boolean per record, bounded to the current physical batch.

    Two float32 forward encodings define the optimizer domain. A computed
    final delta is compared to the representable epsilon bound with NO generic
    physical or floating tolerance: even one-ULP excess must fail closed.
    (We honor epsilon expressed in either Python float or model float32.)
    """
    if final_physical.dtype != np.float64 or clean_physical.dtype != np.float64:
        raise TypeError("Final attack budget requires float64 physical truth")
    if final_physical.shape != clean_physical.shape:
        raise ValueError("Final attack budget requires aligned feature coordinates")
    final_encoded = encode_normalized32(final_physical, feature_names, mean, std)
    clean_encoded = encode_normalized32(clean_physical, feature_names, mean, std)
    if not np.isfinite(final_encoded).all() or not np.isfinite(clean_encoded).all():
        raise ValueError("Budget normalization must produce finite coordinates")
    final_view = final_encoded.reshape(len(final_encoded), -1)
    clean_view = clean_encoded.reshape(len(clean_encoded), -1)
    differences = np.abs(final_view.astype(np.float64) - clean_view.astype(np.float64))
    # Float32 epsilon is the optimization contract's representable upper bound.
    # Do not inflate per-coordinate limits using large-magnitude float32 ULPs.
    declared_limit = max(float(budget.epsilon), float(np.float32(budget.epsilon)))
    if budget.norm == "linf":
        return np.all(differences <= declared_limit, axis=1)
    # This is the globally constrained L2 norm, not per-feature Linf.
    length = np.sqrt(np.einsum('ij,ij->i', differences, differences))
    return length <= declared_limit
