from asmf_lb.engine import ASMFConfig, ASMFEngine
from asmf_lb.models import Frontend, ResourceVector, ServerState


def test_base_score_prefers_less_loaded_server() -> None:
    engine = ASMFEngine(config=ASMFConfig(sample_k=2), seed=1)

    states = {
        "b1": ServerState(
            server_id="b1",
            queue_length=20,
            service_capacity=12.0,
            resources=ResourceVector(cpu=0.8, mem=0.7, io=0.7, net=0.8),
        ),
        "b2": ServerState(
            server_id="b2",
            queue_length=5,
            service_capacity=12.0,
            resources=ResourceVector(cpu=0.3, mem=0.2, io=0.2, net=0.2),
        ),
    }
    frontend = Frontend(frontend_id="f1", allowed_backends=["b1", "b2"])

    engine.update_cache(states.values())
    decision = engine.route(frontend, states)

    assert decision.action == "route"
    assert decision.chosen_backend == "b2"


def test_feedback_penalizes_overload() -> None:
    config = ASMFConfig(eta=0.1)
    engine = ASMFEngine(config=config)

    state = ServerState(
        server_id="b1",
        queue_length=10,
        service_capacity=10.0,
        resources=ResourceVector(cpu=0.5, mem=0.5, io=0.5, net=0.5),
        correction_factor=1.0,
        incoming_rate=14.0,
        service_rate=8.0,
    )

    engine.apply_feedback(state)
    assert state.correction_factor < 1.0


def test_threshold_rejects_when_score_too_low() -> None:
    engine = ASMFEngine(config=ASMFConfig(sample_k=1, threshold=0.9), seed=1)

    states = {
        "b1": ServerState(
            server_id="b1",
            queue_length=30,
            service_capacity=4.0,
            resources=ResourceVector(cpu=0.9, mem=0.9, io=0.9, net=0.9),
        )
    }
    frontend = Frontend(frontend_id="f1", allowed_backends=["b1"])

    engine.update_cache(states.values())
    decision = engine.route(frontend, states)

    assert decision.action == "reject"
    assert decision.chosen_backend is None
