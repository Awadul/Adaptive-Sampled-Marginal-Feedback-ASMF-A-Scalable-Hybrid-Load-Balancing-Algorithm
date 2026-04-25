from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .models import Frontend, RoutingDecision, ServerState


@dataclass
class ASMFConfig:
    sample_k: int = 2
    resource_weights: Tuple[float, float, float, float] = (0.4, 0.25, 0.2, 0.15)
    eta: float = 0.05
    threshold: float = 0.12
    min_correction: float = 0.3
    max_correction: float = 2.5


class ASMFEngine:
    """Adaptive Sampled Marginal Feedback routing engine."""

    def __init__(self, config: Optional[ASMFConfig] = None, seed: Optional[int] = None) -> None:
        self.config = config or ASMFConfig()
        self.rng = random.Random(seed)
        self.score_cache: Dict[str, Tuple[float, int]] = {}

    def update_cache(self, states: Iterable[ServerState]) -> None:
        for st in states:
            base = self._base_score(st)
            corrected = base * st.correction_factor
            self.score_cache[st.server_id] = (max(corrected, 0.0), st.timestamp_ms)

    def apply_feedback(self, state: ServerState) -> None:
        delta = state.incoming_rate - state.service_rate
        next_c = state.correction_factor - self.config.eta * delta
        state.correction_factor = min(max(next_c, self.config.min_correction), self.config.max_correction)

    def route(self, frontend: Frontend, states: Dict[str, ServerState]) -> RoutingDecision:
        allowed = [backend for backend in frontend.allowed_backends if backend in states]
        if not allowed:
            return RoutingDecision(
                frontend_id=frontend.frontend_id,
                chosen_backend=None,
                sampled_backends=[],
                score=0.0,
                action="reject",
            )

        candidates = self._sample_candidates(allowed)
        best_backend, best_score = self._argmax_cached(candidates, states)

        if best_backend is None or best_score < self.config.threshold:
            return RoutingDecision(
                frontend_id=frontend.frontend_id,
                chosen_backend=None,
                sampled_backends=candidates,
                score=best_score,
                action="reject",
            )

        return RoutingDecision(
            frontend_id=frontend.frontend_id,
            chosen_backend=best_backend,
            sampled_backends=candidates,
            score=best_score,
            action="route",
        )

    def route_random(self, frontend: Frontend, states: Dict[str, ServerState]) -> RoutingDecision:
        allowed = [backend for backend in frontend.allowed_backends if backend in states]
        if not allowed:
            return RoutingDecision(frontend.frontend_id, None, [], 0.0, "reject")

        chosen = self.rng.choice(allowed)
        score = self._fresh_score(states[chosen])
        return RoutingDecision(frontend.frontend_id, chosen, [chosen], score, "route")

    def route_least_queue(self, frontend: Frontend, states: Dict[str, ServerState]) -> RoutingDecision:
        allowed = [backend for backend in frontend.allowed_backends if backend in states]
        if not allowed:
            return RoutingDecision(frontend.frontend_id, None, [], 0.0, "reject")

        chosen = min(allowed, key=lambda sid: states[sid].queue_length)
        score = self._fresh_score(states[chosen])
        return RoutingDecision(frontend.frontend_id, chosen, allowed, score, "route")

    def route_power_of_k(self, frontend: Frontend, states: Dict[str, ServerState]) -> RoutingDecision:
        allowed = [backend for backend in frontend.allowed_backends if backend in states]
        if not allowed:
            return RoutingDecision(frontend.frontend_id, None, [], 0.0, "reject")

        candidates = self._sample_candidates(allowed)
        chosen = min(candidates, key=lambda sid: states[sid].queue_length)
        score = self._fresh_score(states[chosen])
        return RoutingDecision(frontend.frontend_id, chosen, candidates, score, "route")

    def route_gmsr(self, frontend: Frontend, states: Dict[str, ServerState]) -> RoutingDecision:
        """Oracle GMSR: full state knowledge, marginal reduction optimization.
        
        GMSR (Generalized Sampling with Marginal Reduction) assumes complete observability
        and chooses the backend that minimizes the maximum queue length across all servers
        (or equivalently, maximizes the marginal reduction in total backlog).
        """
        allowed = [backend for backend in frontend.allowed_backends if backend in states]
        if not allowed:
            return RoutingDecision(frontend.frontend_id, None, [], 0.0, "reject")

        # Full observability: choose the globally least-loaded allowed backend.
        best_backend = min(allowed, key=lambda sid: states[sid].queue_length)

        if best_backend is None:
            return RoutingDecision(frontend.frontend_id, None, allowed, 0.0, "reject")

        score = self._fresh_score(states[best_backend])
        return RoutingDecision(frontend.frontend_id, best_backend, allowed, score, "route")

    def route_asmf_no_sampling(self, frontend: Frontend, states: Dict[str, ServerState]) -> RoutingDecision:
        allowed = [backend for backend in frontend.allowed_backends if backend in states]
        if not allowed:
            return RoutingDecision(frontend.frontend_id, None, [], 0.0, "reject")

        best_backend, best_score = self._argmax_cached(allowed, states)
        if best_backend is None or best_score < self.config.threshold:
            return RoutingDecision(frontend.frontend_id, None, allowed, best_score, "reject")

        return RoutingDecision(frontend.frontend_id, best_backend, allowed, best_score, "route")

    def route_asmf_no_feedback(self, frontend: Frontend, states: Dict[str, ServerState]) -> RoutingDecision:
        return self.route(frontend, states)

    def route_asmf_no_multiresource(self, frontend: Frontend, states: Dict[str, ServerState]) -> RoutingDecision:
        allowed = [backend for backend in frontend.allowed_backends if backend in states]
        if not allowed:
            return RoutingDecision(frontend.frontend_id, None, [], 0.0, "reject")

        candidates = self._sample_candidates(allowed)
        best_backend = None
        best_score = -1.0
        for sid in candidates:
            state = states[sid]
            denominator = 1.0 + state.queue_length
            score = state.service_capacity / max(denominator, 1e-8)
            if score > best_score:
                best_score = score
                best_backend = sid

        if best_backend is None or best_score < self.config.threshold:
            return RoutingDecision(frontend.frontend_id, None, candidates, best_score, "reject")

        return RoutingDecision(frontend.frontend_id, best_backend, candidates, best_score, "route")

    def _sample_candidates(self, allowed: List[str]) -> List[str]:
        k = min(self.config.sample_k, len(allowed))
        return self.rng.sample(allowed, k=k)

    def _argmax_cached(self, candidates: List[str], states: Dict[str, ServerState]) -> Tuple[Optional[str], float]:
        best_backend = None
        best_score = -1.0
        for sid in candidates:
            cached_score = self.score_cache.get(sid)
            if cached_score is None:
                score = self._fresh_score(states[sid])
            else:
                score = cached_score[0]

            if score > best_score:
                best_score = score
                best_backend = sid

        return best_backend, best_score

    def _fresh_score(self, state: ServerState) -> float:
        return self._base_score(state) * state.correction_factor

    def _base_score(self, state: ServerState) -> float:
        load = state.resources.weighted_sum(self.config.resource_weights)
        numerator = state.service_capacity
        denominator = (1.0 + state.queue_length) * (1.0 + load)
        return numerator / max(denominator, 1e-8)
