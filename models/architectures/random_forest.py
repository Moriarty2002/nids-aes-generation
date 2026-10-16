"""Scikit-learn Random Forest registered as a target NIDS plugin.

The default tree/leaf hyperparameters reproduce a reported Hyperopt result on
CICIoT2023 from the 2025 study "Automated Hyperparameter Optimization for
Cyberattack Detection Based on Machine Learning in IoT Systems" (Informatics
12(4), 126; https://doi.org/10.3390/informatics12040126). Those values are a
reproducible starting point for that experimental setting, not universal SOTA
values across datasets, label granularities, splits, or feature schemas.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from core.registry import register_nids


@register_nids("random_forest")
class RandomForestNIDS(RandomForestClassifier):
    """RandomForestClassifier with workflow-compatible input metadata.

    Extending RandomForestClassifier keeps ``fit``, ``predict`` and
    ``predict_proba`` native and preserves scikit-learn estimator semantics.
    The workflow wraps instances with SklearnModelWrapper, while the
    ``requires_flatten`` attribute requests (N, D) input for sequence datasets.
    """

    requires_flatten = True
    REGISTRY_NAME = "random_forest"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_estimators: int = 96,
        *,
        criterion: str = "gini",
        max_depth: int | None = 37,
        min_samples_split: int | float = 23,
        min_samples_leaf: int | float = 5,
        min_weight_fraction_leaf: float = 0.0,
        max_features: str | int | float | None = None,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 1.211795e-5,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int | None = 1,
        random_state: int | None = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: str | dict[int, float] | list[dict[int, float]] | None = None,
        ccp_alpha: float = 0.0,
        max_samples: int | float | None = None,
        monotonic_cst: Any = None,
    ) -> None:
        if int(input_dim) < 1:
            raise ValueError("input_dim must be >= 1")
        if int(num_classes) < 2:
            raise ValueError("num_classes must be >= 2")
        # Retain the model-builder metadata without passing it to sklearn.
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "RandomForestNIDS":
        """Fit after enforcing a two-dimensional numeric feature matrix."""
        if getattr(X, "ndim", None) != 2:
            raise ValueError(f"RandomForestNIDS requires a 2-D array (N, D), got shape {getattr(X, 'shape', None)}")
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {X.shape[1]}")
        if sample_weight is None:
            super().fit(X, y)
        else:
            super().fit(X, y, sample_weight=sample_weight)
        return self
