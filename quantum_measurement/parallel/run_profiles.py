"""Deterministic, backward-compatible schedules for tangent studies.

Profiles decide the configuration of a future independent batch. They never
mutate an active trajectory or alter the derivative map mid-run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Optional


PROFILE_SCHEMA_VERSION = 1
LEGACY_PROFILE = "legacy"
TANGENT_GUARDED_PROFILE = "tangent_guarded_v1"


@dataclass(frozen=True)
class TangentSchedule:
    """Resolved protocol for one independent derivative batch."""

    profile_name: str
    profile_version: int
    gamma: float
    J: float
    dt: float
    n_burnin: int
    n_samples: int
    decision_state: str = "baseline"
    decision_reason: str = "initial schedule"
    parent_fingerprint: Optional[str] = None

    @property
    def n_steps(self) -> int:
        return self.n_burnin + self.n_samples

    @property
    def sampling_time(self) -> float:
        return self.n_samples * self.dt

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["n_steps"] = self.n_steps
        values["sampling_time"] = self.sampling_time
        values["fingerprint"] = self.fingerprint
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TangentSchedule":
        """Reconstruct a schedule from persisted metadata."""

        return cls(
            profile_name=str(values["profile_name"]),
            profile_version=int(values["profile_version"]),
            gamma=float(values["gamma"]),
            J=float(values["J"]),
            dt=float(values["dt"]),
            n_burnin=int(values["n_burnin"]),
            n_samples=int(values["n_samples"]),
            decision_state=str(values.get("decision_state", "baseline")),
            decision_reason=str(values.get("decision_reason", "persisted schedule")),
            parent_fingerprint=values.get("parent_fingerprint"),
        )

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_tangent_schedule(
    *,
    gamma: float,
    J: float,
    dt: float,
    n_burnin: int,
    n_samples: int,
    profile_name: str = LEGACY_PROFILE,
    diagnostics: Optional[Mapping[str, float]] = None,
    max_second_tangent_norm: float = 1.0e3,
    max_second_tangent_sem: float = 10.0,
    dt_min: float = 1.0e-6,
    horizon_reduction: float = 0.5,
) -> TangentSchedule:
    """Resolve a future-batch schedule without changing an active protocol.

    ``legacy`` always returns the supplied schedule unchanged. The guarded
    profile only proposes a refinement after a completed batch exceeds a
    declared tangent-growth or uncertainty threshold. The returned schedule
    has a distinct fingerprint and must be aggregated separately.
    """

    if gamma <= 0.0 or J < 0.0 or dt <= 0.0:
        raise ValueError("gamma and dt must be positive, and J must be nonnegative")
    if n_burnin < 0 or n_samples <= 0:
        raise ValueError("n_burnin must be nonnegative and n_samples positive")
    if profile_name not in {LEGACY_PROFILE, TANGENT_GUARDED_PROFILE}:
        raise ValueError(f"unknown profile: {profile_name}")

    base = TangentSchedule(
        profile_name=profile_name,
        profile_version=PROFILE_SCHEMA_VERSION,
        gamma=float(gamma),
        J=float(J),
        dt=float(dt),
        n_burnin=int(n_burnin),
        n_samples=int(n_samples),
    )
    if profile_name == LEGACY_PROFILE or not diagnostics:
        return base

    max_norm = float(diagnostics.get("max_second_tangent_norm", 0.0))
    second_sem = float(diagnostics.get("d2q_sem", 0.0))
    if max_norm <= max_second_tangent_norm and second_sem <= max_second_tangent_sem:
        return base

    refined_dt = max(float(dt_min), float(dt) * 0.5)
    physical_burnin = n_burnin * dt
    physical_samples = n_samples * dt
    if max_norm > 10.0 * max_second_tangent_norm:
        physical_samples *= float(horizon_reduction)
        reason = "second tangent growth exceeded severe guardrail; refined dt and shortened horizon"
        state = "shortened_horizon"
    else:
        reason = "second tangent growth or SEM exceeded guardrail; refined dt"
        state = "refined_dt"

    refined_burnin = max(0, int(round(physical_burnin / refined_dt)))
    refined_samples = max(1, int(round(physical_samples / refined_dt)))
    return replace(
        base,
        dt=refined_dt,
        n_burnin=refined_burnin,
        n_samples=refined_samples,
        decision_state=state,
        decision_reason=reason,
        parent_fingerprint=base.fingerprint,
    )
