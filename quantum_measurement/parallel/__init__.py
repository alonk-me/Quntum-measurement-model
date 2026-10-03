from .sweep_executor import ParameterSweepExecutor
from .trajectory_worker import TrajectoryConfig, TrajectoryTask
from .validation import ValidationResult, validate_worker0_matches_reference
from .run_profiles import (
    LEGACY_PROFILE,
    PROFILE_SCHEMA_VERSION,
    TANGENT_GUARDED_PROFILE,
    TangentSchedule,
    resolve_tangent_schedule,
)

# Hybrid imports are optional to avoid circular initialization with backend modules.
try:
    from .hybrid_executor import HybridExecutor, HybridExecutorConfig
    from .work_queue import DeterministicWorkQueue, WorkItem, RouteHint
    from .memmap_checkpoint import MemmapCheckpointWriter, CheckpointMetadata
except Exception:  # pragma: no cover - defensive import guard
    HybridExecutor = None
    HybridExecutorConfig = None
    DeterministicWorkQueue = None
    WorkItem = None
    RouteHint = None
    MemmapCheckpointWriter = None
    CheckpointMetadata = None

__all__ = [
    "ParameterSweepExecutor",
    "HybridExecutor",
    "HybridExecutorConfig",
    "DeterministicWorkQueue",
    "WorkItem",
    "RouteHint",
    "MemmapCheckpointWriter",
    "CheckpointMetadata",
    "TrajectoryConfig",
    "TrajectoryTask",
    "ValidationResult",
    "validate_worker0_matches_reference",
    "LEGACY_PROFILE",
    "PROFILE_SCHEMA_VERSION",
    "TANGENT_GUARDED_PROFILE",
    "TangentSchedule",
    "resolve_tangent_schedule",
]
