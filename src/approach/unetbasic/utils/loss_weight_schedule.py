"""Utilities for ramping loss weights over training.

Designed to keep scheduling logic out of the Trainer callback so it stays
small and easy to test.

Config shape (OmegaConf/dict):

loss_weight_schedule:
  enabled: true
  ramps:
    topo_weight: {start_epoch: 10, end_epoch: 40, start_value: 0.0, end_value: 0.15}

All fields are optional per key; missing keys simply won't be scheduled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinearRamp:
    start_epoch: float
    end_epoch: float
    start_value: float
    end_value: float


def _to_float(v: Any, *, name: str) -> float:
    try:
        return float(v)
    except Exception as e:  # pragma: no cover
        raise TypeError(f"{name} must be a number, got {type(v)!r}") from e


def parse_linear_ramp(spec: Mapping[str, Any]) -> LinearRamp:
    """Parse a single ramp spec mapping into a `LinearRamp`."""
    start_epoch = _to_float(spec.get("start_epoch", 0.0), name="start_epoch")
    end_epoch = _to_float(spec.get("end_epoch", start_epoch), name="end_epoch")
    start_value = _to_float(spec.get("start_value", 0.0), name="start_value")
    end_value = _to_float(spec.get("end_value", start_value), name="end_value")

    if end_epoch < start_epoch:
        raise ValueError("end_epoch must be >= start_epoch")

    return LinearRamp(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        start_value=start_value,
        end_value=end_value,
    )


def linear_ramp_value(ramp: LinearRamp, epoch: float) -> float:
    """Compute ramped value at a given epoch (linear interpolation)."""
    if epoch <= ramp.start_epoch:
        return ramp.start_value
    if epoch >= ramp.end_epoch:
        return ramp.end_value

    # If start==end epoch, treat as step to end_value.
    denom = ramp.end_epoch - ramp.start_epoch
    if denom <= 0:
        return ramp.end_value

    t = (epoch - ramp.start_epoch) / denom
    return (1.0 - t) * ramp.start_value + t * ramp.end_value


def compute_scheduled_weights(
    schedule_cfg: Mapping[str, Any] | None,
    *,
    epoch: float,
) -> dict[str, float]:
    """Compute scheduled weights from config at a given epoch.

    Returns only the keys present in schedule_cfg['ramps'].
    """
    if not schedule_cfg:
        return {}

    enabled = bool(schedule_cfg.get("enabled", False))
    if not enabled:
        return {}

    ramps = schedule_cfg.get("ramps", {})
    if ramps is None:
        return {}
    if not isinstance(ramps, Mapping):
        raise TypeError(f"loss_weight_schedule.ramps must be a mapping, got {type(ramps)!r}")

    out: dict[str, float] = {}
    for name, spec in ramps.items():
        if spec is None:
            continue
        if not isinstance(spec, Mapping):
            raise TypeError(f"Ramp spec for {name} must be a mapping, got {type(spec)!r}")

        ramp = parse_linear_ramp(spec)
        out[str(name)] = float(linear_ramp_value(ramp, float(epoch)))

    return out


def apply_weights_to_loss(loss_fn: Any, weights: Mapping[str, float]) -> list[str]:
    """Apply scheduled weights to a loss object by attribute assignment.

    Returns a list of attribute names that were updated.

    Intended for `ThinManifoldLoss`, which stores weights as attributes like
    `topo_weight`, `connectivity_weight`, etc.
    """
    updated: list[str] = []
    for k, v in weights.items():
        if hasattr(loss_fn, k):
            setattr(loss_fn, k, float(v))
            updated.append(k)
    return updated
