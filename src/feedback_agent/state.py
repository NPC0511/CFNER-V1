"""Structured observations and actions for the feedback agent."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OldClassState:
    entity_type: str
    f1_before: Optional[float] = None
    f1_current: Optional[float] = None
    f1_drop: float = 0.0
    feature_drift: float = 0.0
    prototype_similarity: Optional[float] = None
    teacher_confusion: Optional[float] = None
    pseudo_confidence: Optional[float] = None
    pseudo_entropy: Optional[float] = None
    coverage: Dict[str, int] = field(default_factory=dict)
    active_actions: List[str] = field(default_factory=list)


@dataclass
class TrainingState:
    task_id: int
    step: int = 0
    domain: str = ""
    new_types: List[str] = field(default_factory=list)
    old_types: List[str] = field(default_factory=list)
    seen_types: List[str] = field(default_factory=list)
    old_classes: Dict[str, OldClassState] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    observation_index: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class ActionRequest:
    action: str
    step: int
    targets: List[str] = field(default_factory=list)
    delta: float = 0.0
    duration_steps: int = 0
    cooldown_steps: int = 0
    trigger_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRecord:
    request: ActionRequest
    accepted: bool
    reason: str
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    rollback: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class TaskReflection:
    task_id: int
    summary_tags: List[str] = field(default_factory=list)
    next_task_focus: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    action_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
