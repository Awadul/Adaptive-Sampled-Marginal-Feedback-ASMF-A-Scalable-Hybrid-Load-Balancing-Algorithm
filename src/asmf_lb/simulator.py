from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from .engine import ASMFEngine
from .models import Frontend, Job, ResourceVector, ServerState, SimulationMetrics


@dataclass
class SimulationConfig:
    duration_ms: int = 180_000
    time_step_ms: int = 100
    arrivals_per_step: float = 7.0
    stale_update_interval_ms: int = 300
    reject_instead_of_delay: bool = True
    max_delay_buffer: int = 500
    workload_type: str = "poisson"
    burst_on_multiplier: float = 2.5
    burst_off_multiplier: float = 0.4
    burst_period_steps: int = 60
    zipf_alpha: float = 1.2
    hotspot_fraction: float = 0.2
    backend_failure_at_ms: int = -1
    failure_fraction: float = 0.0
    load_spike_at_ms: int = -1
    load_spike_multiplier: float = 1.0
    load_spike_duration_ms: int = 0
    degrade_at_ms: int = -1
    degrade_factor: float = 1.0
    trace_interval_ms: int = 1000
    convergence_window_points: int = 12
    convergence_tolerance: float = 0.05
    capture_diagnostics: bool = True


class LoadBalancingSimulator:
    def __init__(
        self,
        frontends: List[Frontend],
        initial_states: Dict[str, ServerState],
        config: SimulationConfig,
        seed: int = 42,
    ) -> None:
        self.frontends = frontends
        self.initial_states = initial_states
        self.config = config
        self.rng = random.Random(seed)

    def run(self, policy: str, engine: ASMFEngine) -> SimulationMetrics:
        states = self._clone_states()
        metrics = SimulationMetrics(policy=policy)
        delayed_jobs: List[Job] = []
        failed_backends: Set[str] = set()
        scale_degraded = False
        convergence_found = False
        queue_trace: List[float] = []

        next_job_id = 0
        now_ms = 0
        while now_ms < self.config.duration_ms:
            self._apply_events(states, now_ms, failed_backends, scale_degraded)
            if self.config.degrade_at_ms >= 0 and now_ms >= self.config.degrade_at_ms:
                scale_degraded = True

            if now_ms % self.config.stale_update_interval_ms == 0:
                engine.update_cache(states.values())
                metrics.state_updates_sent += len(states)
                metrics.bytes_transferred_est += len(states) * 64

            self._service_step(states, metrics, self.config.time_step_ms)

            for st in states.values():
                st.timestamp_ms = now_ms
                if policy != "asmf_no_feedback":
                    engine.apply_feedback(st)

            arrivals = self._arrival_count(now_ms)
            for _ in range(arrivals):
                frontend = self._choose_frontend(states)
                job = Job(
                    job_id=next_job_id,
                    frontend_id=frontend.frontend_id,
                    arrival_ms=now_ms,
                    service_demand=max(0.1, self.rng.expovariate(1.0 / 1.5)),
                )
                next_job_id += 1
                metrics.jobs_generated += 1

                decision = self._dispatch(policy, engine, frontend, states)
                qcount = self._query_count(policy, frontend, decision, states)
                metrics.state_queries += qcount
                metrics.bytes_transferred_est += qcount * 48
                if decision.action == "route" and decision.chosen_backend is not None:
                    self._enqueue(states[decision.chosen_backend], job)
                    metrics.jobs_routed += 1
                else:
                    if self.config.reject_instead_of_delay or len(delayed_jobs) >= self.config.max_delay_buffer:
                        metrics.jobs_rejected += 1
                    else:
                        metrics.jobs_delayed += 1
                        delayed_jobs.append(job)

            # Retry delayed jobs opportunistically.
            retried = []
            for job in delayed_jobs:
                frontend = next(f for f in self.frontends if f.frontend_id == job.frontend_id)
                decision = self._dispatch(policy, engine, frontend, states)
                qcount = self._query_count(policy, frontend, decision, states)
                metrics.state_queries += qcount
                metrics.bytes_transferred_est += qcount * 48
                if decision.action == "route" and decision.chosen_backend is not None:
                    self._enqueue(states[decision.chosen_backend], job)
                    metrics.jobs_routed += 1
                else:
                    retried.append(job)
            delayed_jobs = retried

            total_queue = sum(st.queue_length for st in states.values())
            metrics.backlog_area += total_queue * (self.config.time_step_ms / 1000.0)
            metrics.max_queue_observed = max(metrics.max_queue_observed, total_queue)
            if self.config.capture_diagnostics:
                metrics.queue_snapshots[now_ms] = {sid: st.queue_length for sid, st in states.items()}

            if self.config.capture_diagnostics and now_ms % max(self.config.trace_interval_ms, self.config.time_step_ms) == 0:
                queue_values = [st.queue_length for st in states.values()]
                queue_mean = float(sum(queue_values) / max(len(queue_values), 1))
                queue_var = float(np.var(queue_values)) if queue_values else 0.0
                rejection_rate = metrics.jobs_rejected / max(metrics.jobs_generated, 1)
                metrics.trace_points += 1
                metrics.trace_sum_queue_mean += queue_mean
                metrics.trace_sum_queue_var += queue_var
                metrics.trace_sum_rejection_rate += rejection_rate
                metrics.trace_records.append(
                    {
                        "time_ms": float(now_ms),
                        "queue_mean": queue_mean,
                        "queue_var": queue_var,
                        "rejection_rate": rejection_rate,
                    }
                )
                queue_trace.append(queue_mean)

                if not convergence_found and len(queue_trace) >= self.config.convergence_window_points:
                    window = queue_trace[-self.config.convergence_window_points :]
                    mean_w = sum(window) / len(window)
                    max_dev = max(abs(v - mean_w) for v in window) / max(abs(mean_w), 1.0)
                    if max_dev <= self.config.convergence_tolerance:
                        metrics.convergence_time_ms = now_ms
                        convergence_found = True

            now_ms += self.config.time_step_ms

        if metrics.convergence_time_ms == 0:
            metrics.convergence_time_ms = self.config.duration_ms

        if self.config.capture_diagnostics and len(queue_trace) > 1:
            metrics.mean_queue_variance = float(np.var(queue_trace))
            deltas = [abs(queue_trace[i] - queue_trace[i - 1]) for i in range(1, len(queue_trace))]
            metrics.oscillation_index = float(sum(deltas) / len(deltas))

        return metrics

    def _dispatch(
        self,
        policy: str,
        engine: ASMFEngine,
        frontend: Frontend,
        states: Dict[str, ServerState],
    ):
        if policy == "asmf":
            return engine.route(frontend, states)
        if policy == "asmf_no_feedback":
            return engine.route_asmf_no_feedback(frontend, states)
        if policy == "asmf_no_sampling":
            return engine.route_asmf_no_sampling(frontend, states)
        if policy == "asmf_no_multiresource":
            return engine.route_asmf_no_multiresource(frontend, states)
        if policy == "random":
            return engine.route_random(frontend, states)
        if policy == "least_queue":
            return engine.route_least_queue(frontend, states)
        if policy == "p2c":
            return engine.route_power_of_k(frontend, states)
        if policy == "gmsr":
            return engine.route_gmsr(frontend, states)
        raise ValueError(f"Unknown policy: {policy}")

    def _query_count(self, policy: str, frontend: Frontend, decision, states: Dict[str, ServerState]) -> int:
        allowed_count = len([sid for sid in frontend.allowed_backends if sid in states])
        if policy in {"gmsr", "least_queue", "asmf_no_sampling"}:
            return allowed_count
        if policy == "random":
            return 1
        return max(len(decision.sampled_backends), 1)

    def _arrival_count(self, now_ms: int) -> int:
        lam = self.config.arrivals_per_step
        step_idx = now_ms // max(self.config.time_step_ms, 1)

        if self.config.workload_type == "bursty":
            period = max(self.config.burst_period_steps, 1)
            in_on = (step_idx // period) % 2 == 0
            lam *= self.config.burst_on_multiplier if in_on else self.config.burst_off_multiplier

        if self.config.load_spike_at_ms >= 0:
            spike_end = self.config.load_spike_at_ms + max(self.config.load_spike_duration_ms, 0)
            if self.config.load_spike_at_ms <= now_ms < spike_end:
                lam *= self.config.load_spike_multiplier

        return self._poisson(max(lam, 0.0))

    def _choose_frontend(self, states: Dict[str, ServerState]) -> Frontend:
        if self.config.workload_type != "zipf_skew":
            return self.rng.choice(self.frontends)

        fronts = self.frontends
        n = len(fronts)
        if n == 0:
            raise ValueError("No frontends available")

        weights = [1.0 / ((i + 1) ** max(self.config.zipf_alpha, 0.1)) for i in range(n)]
        total_w = sum(weights)
        r = self.rng.random() * total_w
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                return fronts[i]
        return fronts[-1]

    def _apply_events(
        self,
        states: Dict[str, ServerState],
        now_ms: int,
        failed_backends: Set[str],
        scale_degraded: bool,
    ) -> None:
        if self.config.backend_failure_at_ms >= 0 and now_ms == self.config.backend_failure_at_ms and self.config.failure_fraction > 0.0:
            candidates = [sid for sid in states if sid not in failed_backends]
            k = max(1, int(len(candidates) * self.config.failure_fraction))
            for sid in self.rng.sample(candidates, k=min(k, len(candidates))):
                failed_backends.add(sid)
                states[sid].service_capacity = 0.0

        if self.config.degrade_at_ms >= 0 and now_ms == self.config.degrade_at_ms and not scale_degraded:
            factor = max(min(self.config.degrade_factor, 1.0), 0.05)
            for sid in states:
                states[sid].service_capacity *= factor

    def _service_step(
        self,
        states: Dict[str, ServerState],
        metrics: SimulationMetrics,
        dt_ms: int,
    ) -> None:
        dt_seconds = dt_ms / 1000.0
        for st in states.values():
            arrivals = st.incoming_rate
            capacity_jobs = st.service_capacity * dt_seconds
            served = min(st.queue_length, int(math.floor(capacity_jobs)))
            st.queue_length -= served
            st.service_rate = served / max(dt_seconds, 1e-9)
            metrics.completed_jobs += served
            metrics.total_wait_time += st.queue_length * dt_seconds

            # Synthetic resource pressure follows queue pressure with noise.
            pressure = min(1.0, st.queue_length / 80.0)
            st.resources = ResourceVector(
                cpu=min(1.0, 0.35 + 0.55 * pressure + self.rng.uniform(-0.05, 0.05)),
                mem=min(1.0, 0.30 + 0.45 * pressure + self.rng.uniform(-0.05, 0.05)),
                io=min(1.0, 0.25 + 0.40 * pressure + self.rng.uniform(-0.05, 0.05)),
                net=min(1.0, 0.20 + 0.50 * pressure + self.rng.uniform(-0.05, 0.05)),
            )
            st.incoming_rate = max(0.0, arrivals * 0.5)

    def _enqueue(self, state: ServerState, job: Job) -> None:
        state.queue_length += max(1, int(math.ceil(job.service_demand)))
        state.incoming_rate += 1.0

    def _clone_states(self) -> Dict[str, ServerState]:
        return {
            sid: ServerState(
                server_id=st.server_id,
                queue_length=st.queue_length,
                service_capacity=st.service_capacity,
                resources=ResourceVector(
                    cpu=st.resources.cpu,
                    mem=st.resources.mem,
                    io=st.resources.io,
                    net=st.resources.net,
                ),
                incoming_rate=st.incoming_rate,
                service_rate=st.service_rate,
                correction_factor=st.correction_factor,
                timestamp_ms=st.timestamp_ms,
            )
            for sid, st in self.initial_states.items()
        }

    def _poisson(self, lam: float) -> int:
        l = math.exp(-lam)
        k = 0
        p = 1.0
        while p > l:
            k += 1
            p *= self.rng.random()
        return k - 1


def build_default_topology(num_frontends: int = 4, num_backends: int = 12) -> Tuple[List[Frontend], Dict[str, ServerState]]:
    rng = random.Random(7)

    backends = [f"b{i+1}" for i in range(num_backends)]
    states: Dict[str, ServerState] = {}
    for sid in backends:
        states[sid] = ServerState(
            server_id=sid,
            queue_length=rng.randint(1, 8),
            service_capacity=rng.uniform(10.0, 20.0),
            resources=ResourceVector(
                cpu=rng.uniform(0.25, 0.55),
                mem=rng.uniform(0.20, 0.50),
                io=rng.uniform(0.15, 0.45),
                net=rng.uniform(0.15, 0.50),
            ),
        )

    frontends: List[Frontend] = []
    for i in range(num_frontends):
        allowed = rng.sample(backends, k=max(4, int(num_backends * 0.66)))
        frontends.append(Frontend(frontend_id=f"f{i+1}", allowed_backends=allowed))

    return frontends, states
