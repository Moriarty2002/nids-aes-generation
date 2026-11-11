"""Independent, scoped accounting of *observed* model operations.

A query is an evaluated example, not an optimizer iteration. Calls and example
counts are both retained; a batched call of B records costs B example queries.
The distinction between inference and gradient-bearing model forwards prevents
comparing PGD gradient access with black-box score queries as though equivalent.

Only model operations inside a scope are charged. Evaluation-target scopes are
never visible to attack plugins. Unsupported attack model interfaces fail closed.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import torch


def _batch_size(x: Any) -> int:
    try:
        size = int(len(x))
    except (TypeError, ValueError) as exc:
        raise ValueError('Model-query accounting requires batched inputs') from exc
    if size < 0:
        raise ValueError('Invalid model-query batch size')
    return size


@dataclass
class ModelQueryCounts:
    inference_calls: int = 0
    inference_records: int = 0
    gradient_forward_calls: int = 0
    gradient_forward_records: int = 0
    feature_calls: int = 0
    feature_records: int = 0

    def add(self, other: 'ModelQueryCounts') -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def metrics(self, prefix: str) -> dict[str, float]:
        return {f'{prefix}_{name}': float(getattr(self, name)) for name in self.__dataclass_fields__}


class ModelQueryMeter(AbstractContextManager['ModelQueryMeter']):
    """Measure model-level calls made within one synchronous evaluator phase.

    Torch: root module forward-pre-hook catches both wrapper forwards and direct
    module invocations made by gradient attacks; no duplicate wrapper counting.
    Sklearn/custom adapters: scoped predict/predict_proba instrumentation.
    Inference to feature extraction is a distinct category, not target logits.
    """

    def __init__(self, wrapper: Any, *, forbidden: bool = False) -> None:
        self.wrapper = wrapper
        self.forbidden = forbidden
        self.counts = ModelQueryCounts()
        self._handle: Any = None
        self._feature_depth = 0
        self._original_methods: dict[str, tuple[Any, bool]] = {}

    def _record(self, kind: str, count: int) -> None:
        if self.forbidden:
            raise RuntimeError('Frozen evaluation target cannot be queried during attack generation')
        attr = 'gradient_forward' if kind == 'gradient' else kind
        setattr(self.counts, f'{attr}_calls', getattr(self.counts, f'{attr}_calls') + 1)
        setattr(self.counts, f'{attr}_records', getattr(self.counts, f'{attr}_records') + count)

    def __enter__(self) -> 'ModelQueryMeter':
        if self.wrapper is None:
            return self
        module = getattr(self.wrapper, 'model', None)
        if isinstance(module, torch.nn.Module):
            def hook(_module: Any, args: tuple[Any, ...]) -> None:
                if not args:
                    raise RuntimeError('Cannot meter a model forward without batched input')
                x = args[0]
                gradient = bool(torch.is_grad_enabled() and isinstance(x, torch.Tensor) and x.requires_grad)
                kind = 'feature' if self._feature_depth else ('gradient' if gradient else 'inference')
                self._record(kind, _batch_size(x))
            self._handle = module.register_forward_pre_hook(hook)
            # Direct calls to wrapper.feature extraction may bypass root forward.
            methods = (
                'predict', 'predict_proba', 'logits', 'predict_logits', 'get_logits',
                'extract_features', 'forward_differentiable', 'get_gradients', 'loss',
            ) if self.forbidden else ('extract_features',)
        else:
            methods = (
                'predict', 'predict_proba', 'logits', 'predict_logits', 'get_logits',
                'extract_features', 'forward_differentiable', 'get_gradients', 'loss',
            )
        for name in methods:
            method = getattr(self.wrapper, name, None)
            if not callable(method):
                continue
            if name in ('get_gradients', 'forward_differentiable', 'loss'):
                kind = 'gradient'
            elif name == 'extract_features':
                kind = 'feature'
            else:
                kind = 'inference'
            @wraps(method)
            def instrumented(x: Any, *args: Any, _method=method, _kind=kind, **kwargs: Any) -> Any:
                if isinstance(module, torch.nn.Module) and _kind == 'feature' and not self.forbidden:
                    # PyTorch feature extraction normally performs one real root
                    # forward. Attribute *that* forward to the detector instead
                    # of double-counting it as an inference and a feature call.
                    before = self.counts.feature_calls
                    self._feature_depth += 1
                    try:
                        result = _method(x, *args, **kwargs)
                    finally:
                        self._feature_depth -= 1
                    if self.counts.feature_calls == before:
                        # Custom unheaded extractors may bypass root forward.
                        self._record('feature', _batch_size(x))
                    return result
                self._record(_kind, _batch_size(x))
                return _method(x, *args, **kwargs)
            self._original_methods[name] = (method, name in vars(self.wrapper))
            try:
                setattr(self.wrapper, name, instrumented)
            except (AttributeError, TypeError) as exc:
                self.__exit__(None, None, None)
                raise RuntimeError(f'Cannot independently meter model method {name}') from exc
        # An opaque binding that exposes no oracle methods cannot be queried.
        # Do not invent query costs for a model-independent test double.
        return self

    def __exit__(self, *_: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        for name, (original, preexisting) in self._original_methods.items():
            if preexisting:
                setattr(self.wrapper, name, original)
            else:
                delattr(self.wrapper, name)
        self._original_methods.clear()
