from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

        next_job_id = 0
        now_ms = 0
        while now_ms < self.config.duration_ms:
            if now_ms % self.config.stale_update_interval_ms == 0:
                engine.update_cache(states.values())

            self._service_step(states, metrics, self.config.time_step_ms)

            for st in states.values():
                st.timestamp_ms = now_ms
                engine.apply_feedback(st)

            arrivals = self._poisson(self.config.arrivals_per_step)
            for _ in range(arrivals):
                frontend = self.rng.choice(self.frontends)
                job = Job(
                    job_id=next_job_id,
                    frontend_id=frontend.frontend_id,
                    arrival_ms=now_ms,
                    service_demand=max(0.1, self.rng.expovariate(1.0 / 1.5)),
                )
                next_job_id += 1
                metrics.jobs_generated += 1

                decision = self._dispatch(policy, engine, frontend, states)
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
                if decision.action == "route" and decision.chosen_backend is not None:
                    self._enqueue(states[decision.chosen_backend], job)
                    metrics.jobs_routed += 1
                else:
                    retried.append(job)
            delayed_jobs = retried

            total_queue = sum(st.queue_length for st in states.values())
            metrics.backlog_area += total_queue * (self.config.time_step_ms / 1000.0)
            metrics.max_queue_observed = max(metrics.max_queue_observed, total_queue)
            metrics.queue_snapshots[now_ms] = {sid: st.queue_length for sid, st in states.items()}
            now_ms += self.config.time_step_ms

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
        if policy == "random":
            return engine.route_random(frontend, states)
        if policy == "least_queue":
            return engine.route_least_queue(frontend, states)
        if policy == "p2c":
            return engine.route_power_of_k(frontend, states)
        raise ValueError(f"Unknown policy: {policy}")

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
