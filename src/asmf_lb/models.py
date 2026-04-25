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
    state_updates_sent: int = 0
    state_queries: int = 0
    bytes_transferred_est: int = 0
    convergence_time_ms: int = 0
    mean_queue_variance: float = 0.0
    oscillation_index: float = 0.0
    trace_points: int = 0
    trace_sum_queue_mean: float = 0.0
    trace_sum_queue_var: float = 0.0
    trace_sum_rejection_rate: float = 0.0
    trace_records: List[Dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        throughput = self.completed_jobs / max(self.jobs_generated, 1)
        acceptance = self.jobs_routed / max(self.jobs_generated, 1)
        rejection = self.jobs_rejected / max(self.jobs_generated, 1)
        avg_wait = self.total_wait_time / max(self.completed_jobs, 1)
        queries_per_decision = self.state_queries / max(self.jobs_generated, 1)
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
            "state_updates_sent": self.state_updates_sent,
            "state_queries": self.state_queries,
            "bytes_transferred_est": self.bytes_transferred_est,
            "queries_per_decision": queries_per_decision,
            "convergence_time_ms": self.convergence_time_ms,
            "mean_queue_variance": self.mean_queue_variance,
            "oscillation_index": self.oscillation_index,
            "trace_points": self.trace_points,
            "trace_queue_mean_avg": self.trace_sum_queue_mean / max(self.trace_points, 1),
            "trace_queue_var_avg": self.trace_sum_queue_var / max(self.trace_points, 1),
            "trace_rejection_rate_avg": self.trace_sum_rejection_rate / max(self.trace_points, 1),
        }
