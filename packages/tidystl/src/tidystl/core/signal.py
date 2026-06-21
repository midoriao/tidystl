from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray


@dataclass
class Signal:
    """A batch of multi-variable traces on a shared time grid.

    Attributes:
        values: float array of shape `(N, S, T)`; `N` traces, `S` variables,
            `T` timesteps.
        times: float array of shape `(T,)`, strictly increasing, in physical
            time units. Temporal intervals such as `G[0,10]` refer to these
            units, not to sample indices.
        labels: mapping from variable name to its index on the `S` axis.

    Indexing: `sig["x"]` returns the `(N, T)` array for one variable;
    `sig["x", 3]` returns the `(N,)` values at timestep index 3. An unknown
    variable name raises `KeyError`. Build directly with
    `Signal(values=..., times=..., labels=...)` for an already-stacked
    `(N, S, T)` array, or with `Signal.from_dict(...)`.
    """

    values: NDArray[np.floating]  # (N, S, T)
    times: NDArray[np.floating]  # (T,)
    labels: dict[str, int]  # variable name -> S-axis index

    @classmethod
    def from_dict(
        cls,
        times: NDArray[np.floating],
        values: dict[str, NDArray[np.floating]],
    ) -> Signal:
        """Build a `Signal` from a `{name: array}` mapping on a time grid.

        Each value array is `(N, T)`, or `(T,)` which is auto-expanded to
        `(1, T)`. All variables must share batch size `N` and timestep count
        `T`, and `T` must equal `times.shape[0]`. A timestep mismatch raises
        `ValueError`.
        """
        t_len = times.shape[0]
        arrays: list[NDArray[np.floating]] = []
        labels: dict[str, int] = {}
        for i, (name, arr) in enumerate(values.items()):
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]  # (T,) -> (1, T)
            if arr.shape[-1] != t_len:
                raise ValueError(
                    f"values[{name!r}] has {arr.shape[-1]} timesteps, but times has {t_len}"
                )
            arrays.append(arr)
            labels[name] = i
        stacked = np.stack(arrays, axis=1)  # (N, S, T)
        return cls(values=stacked, times=times, labels=labels)

    @overload
    def __getitem__(self, key: str) -> NDArray[np.floating]: ...
    @overload
    def __getitem__(self, key: tuple[str, int]) -> NDArray[np.floating]: ...

    def __getitem__(self, key: str | tuple[str, int]) -> NDArray[np.floating]:
        if isinstance(key, str):
            idx = self.labels[key]  # raises KeyError if missing
            return self.values[:, idx, :]  # (N, T)
        name, t = key
        idx = self.labels[name]
        return self.values[:, idx, t]  # (N,)


@dataclass
class TorchSignal:
    """Batched multi-variable time series backed by torch tensors. Shape: (N, S, T)."""

    values: Any
    times: NDArray[np.floating]
    labels: dict[str, int]

    @classmethod
    def from_dict(
        cls,
        times: NDArray[np.floating],
        values: dict[str, Any],
    ) -> TorchSignal:
        """Build a `TorchSignal` from a `{name: tensor}` mapping.

        The torch analogue of `Signal.from_dict`: each value tensor is
        `(N, T)` or `(T,)` (auto-expanded to `(1, T)`); all variables must
        share `N` and `T`, and `T` must equal `times.shape[0]`. A timestep
        mismatch raises `ValueError`.
        """
        import torch  # type: ignore[import-untyped]

        _t: Any = torch  # shadow as Any so downstream calls are not Unknown
        t_len = times.shape[0]
        arrays: list[Any] = []
        labels: dict[str, int] = {}
        for i, (name, arr) in enumerate(values.items()):
            if arr.ndim == 1:
                arr = arr.unsqueeze(0)
            if arr.shape[-1] != t_len:
                raise ValueError(
                    f"values[{name!r}] has {arr.shape[-1]} timesteps, but times has {t_len}"
                )
            arrays.append(arr)
            labels[name] = i
        stacked = _t.stack(arrays, dim=1)
        return cls(values=stacked, times=times, labels=labels)

    @overload
    def __getitem__(self, key: str) -> Any: ...
    @overload
    def __getitem__(self, key: tuple[str, int]) -> Any: ...

    def __getitem__(self, key: str | tuple[str, int]) -> Any:
        if isinstance(key, str):
            idx = self.labels[key]
            return self.values[:, idx, :]
        name, t = key
        idx = self.labels[name]
        return self.values[:, idx, t]
