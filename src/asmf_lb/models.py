from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ResourceVector:
    cpu: float
    mem: float
    io: float
    net: float

    def weighted_sum(self, weights: Tuple[float, float, float, float]) -> float:
        a, b, g, d = weights
        return a * self.cpu + b * self.mem + g * self.io + d * self.net


@dataclass
class ServerState:
    server_id: str
    queue_length: int
    service_capacity: float
    resources: ResourceVector
    incoming_rate: float = 0.0
    service_rate: float = 0.0
    correction_factor: float = 1.0
    timestamp_ms: int = 0


@dataclass
class Job:
    job_id: int
    frontend_id: str
    arrival_ms: int
    service_demand: float


@dataclass
class Frontend:
    frontend_id: str
    allowed_backends: List[str]


@dataclass
class RoutingDecision:
    frontend_id: str
    chosen_backend: Optional[str]
    sampled_backends: List[str]
    score: float
    action: str


@dataclass
class SimulationMetrics:
    policy: str
    jobs_generated: int = 0
    jobs_routed: int = 0
    jobs_rejected: int = 0
    jobs_delayed: int = 0
    completed_jobs: int = 0
    total_wait_time: float = 0.0
    backlog_area: float = 0.0
    max_queue_observed: int = 0
    queue_snapshots: Dict[int, Dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        throughput = self.completed_jobs / max(self.jobs_generated, 1)
        acceptance = self.jobs_routed / max(self.jobs_generated, 1)
        rejection = self.jobs_rejected / max(self.jobs_generated, 1)
        avg_wait = self.total_wait_time / max(self.completed_jobs, 1)
        return {
            "policy": self.policy,
            "jobs_generated": self.jobs_generated,
            "jobs_routed": self.jobs_routed,
            "jobs_rejected": self.jobs_rejected,
            "jobs_delayed": self.jobs_delayed,
            "completed_jobs": self.completed_jobs,
            "throughput": throughput,
            "acceptance_rate": acceptance,
            "rejection_rate": rejection,
            "avg_wait_time": avg_wait,
            "max_queue_observed": self.max_queue_observed,
            "backlog_area": self.backlog_area,
        }
