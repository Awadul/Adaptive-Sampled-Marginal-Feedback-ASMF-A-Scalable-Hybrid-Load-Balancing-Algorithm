from __future__ import annotations

from asmf_lb.engine import ASMFConfig, ASMFEngine
from asmf_lb.simulator import LoadBalancingSimulator, SimulationConfig, build_default_topology


def main() -> None:
    frontends, states = build_default_topology(num_frontends=3, num_backends=8)
    simulator = LoadBalancingSimulator(
        frontends,
        states,
        SimulationConfig(duration_ms=20_000, arrivals_per_step=5.0),
        seed=5,
    )

    engine = ASMFEngine(config=ASMFConfig(sample_k=3, threshold=0.08), seed=5)
    metrics = simulator.run(policy="asmf", engine=engine)
    print(metrics.as_dict())


if __name__ == "__main__":
    main()
