from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from asmf_lb.engine import ASMFConfig, ASMFEngine
from asmf_lb.experiments import _default_scenarios
from asmf_lb.simulator import LoadBalancingSimulator, SimulationConfig, build_default_topology


def _parse_seeds(seed_spec: str) -> List[int]:
    if ":" in seed_spec:
        start, end = seed_spec.split(":", maxsplit=1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in seed_spec.split(",") if x.strip()]


def _sim_config_from_scenario(
    scenario: Dict[str, Any],
    duration_scale: float = 1.0,
    capture_diagnostics: bool = False,
) -> SimulationConfig:
    scaled_duration = int(max(10_000, int(scenario["duration_ms"]) * duration_scale))
    return SimulationConfig(
        duration_ms=scaled_duration,
        time_step_ms=int(scenario.get("time_step_ms", 100)),
        arrivals_per_step=float(scenario["arrivals_per_step"]),
        stale_update_interval_ms=int(scenario["stale_update_interval_ms"]),
        reject_instead_of_delay=bool(scenario.get("reject_instead_of_delay", True)),
        max_delay_buffer=int(scenario.get("max_delay_buffer", 500)),
        workload_type=str(scenario.get("workload_type", "poisson")),
        burst_on_multiplier=float(scenario.get("burst_on_multiplier", 2.5)),
        burst_off_multiplier=float(scenario.get("burst_off_multiplier", 0.4)),
        burst_period_steps=int(scenario.get("burst_period_steps", 60)),
        zipf_alpha=float(scenario.get("zipf_alpha", 1.2)),
        hotspot_fraction=float(scenario.get("hotspot_fraction", 0.2)),
        backend_failure_at_ms=int(scenario.get("backend_failure_at_ms", -1)),
        failure_fraction=float(scenario.get("failure_fraction", 0.0)),
        load_spike_at_ms=int(scenario.get("load_spike_at_ms", -1)),
        load_spike_multiplier=float(scenario.get("load_spike_multiplier", 1.0)),
        load_spike_duration_ms=int(scenario.get("load_spike_duration_ms", 0)),
        degrade_at_ms=int(scenario.get("degrade_at_ms", -1)),
        degrade_factor=float(scenario.get("degrade_factor", 1.0)),
        trace_interval_ms=int(scenario.get("trace_interval_ms", 1000)),
        capture_diagnostics=bool(capture_diagnostics),
    )


def _engine_config_from_scenario(scenario: Dict[str, Any]) -> ASMFConfig:
    return ASMFConfig(
        sample_k=int(scenario["sample_k"]),
        resource_weights=tuple(scenario["resource_weights"]),
        eta=float(scenario["eta"]),
        threshold=float(scenario["threshold"]),
        min_correction=float(scenario.get("min_correction", 0.3)),
        max_correction=float(scenario.get("max_correction", 2.5)),
    )


def _run_policy(
    scenario: Dict[str, Any],
    seed: int,
    policy_name: str,
    dispatch_policy: str,
    cfg: ASMFConfig,
    duration_scale: float,
    capture_diagnostics: bool,
) -> Dict[str, Any]:
    sim_cfg = _sim_config_from_scenario(
        scenario,
        duration_scale=duration_scale,
        capture_diagnostics=capture_diagnostics,
    )
    frontends, states = build_default_topology(
        num_frontends=int(scenario["num_frontends"]),
        num_backends=int(scenario["num_backends"]),
    )
    simulator = LoadBalancingSimulator(frontends, states, sim_cfg, seed=seed)
    engine = ASMFEngine(config=cfg, seed=seed)
    row = simulator.run(policy=dispatch_policy, engine=engine).as_dict()
    row["policy"] = policy_name
    row["seed"] = seed
    row["scenario"] = scenario["name"]
    row["duration_ms"] = sim_cfg.duration_ms
    row["time_step_ms"] = sim_cfg.time_step_ms
    row["arrivals_per_step"] = sim_cfg.arrivals_per_step
    row["stale_update_interval_ms"] = sim_cfg.stale_update_interval_ms
    row["workload_type"] = sim_cfg.workload_type
    row["num_frontends"] = scenario["num_frontends"]
    row["num_backends"] = scenario["num_backends"]
    row["sample_k"] = cfg.sample_k
    row["threshold"] = cfg.threshold
    row["eta"] = cfg.eta
    row["failure_fraction"] = sim_cfg.failure_fraction
    row["load_spike_multiplier"] = sim_cfg.load_spike_multiplier
    row["degrade_factor"] = sim_cfg.degrade_factor
    return row


def _run_task(task: Tuple[Dict[str, Any], int, str, str, Dict[str, Any], float, bool]) -> Dict[str, Any]:
    scenario, seed, policy_name, dispatch_policy, cfg_dict, duration_scale, capture_diagnostics = task
    cfg = ASMFConfig(**cfg_dict)
    return _run_policy(
        scenario=scenario,
        seed=seed,
        policy_name=policy_name,
        dispatch_policy=dispatch_policy,
        cfg=cfg,
        duration_scale=duration_scale,
        capture_diagnostics=capture_diagnostics,
    )


def _pairwise_vs_base(df: pd.DataFrame, base: str = "asmf_default") -> pd.DataFrame:
    id_cols = ["scenario", "seed"]
    metrics = ["throughput", "avg_wait_time", "backlog_area", "queries_per_decision"]

    base_df = df[df["policy"] == base][id_cols + metrics].copy()
    base_df.columns = id_cols + [f"{m}_base" for m in metrics]

    rows: List[Dict[str, Any]] = []
    for pol in sorted(df["policy"].unique()):
        if pol == base:
            continue
        pol_df = df[df["policy"] == pol][id_cols + metrics].copy()
        pol_df.columns = id_cols + [f"{m}_pol" for m in metrics]
        merged = pol_df.merge(base_df, on=id_cols, how="inner")
        if merged.empty:
            continue
        rows.append(
            {
                "policy": pol,
                "n": len(merged),
                "throughput_gain_pct": float(
                    100.0
                    * ((merged["throughput_pol"] - merged["throughput_base"]) / merged["throughput_base"]).mean()
                ),
                "wait_reduction_pct": float(
                    100.0
                    * ((merged["avg_wait_time_base"] - merged["avg_wait_time_pol"]) / merged["avg_wait_time_base"]).mean()
                ),
                "backlog_reduction_pct": float(
                    100.0
                    * ((merged["backlog_area_base"] - merged["backlog_area_pol"]) / merged["backlog_area_base"]).mean()
                ),
                "qpd_reduction_pct": float(
                    100.0
                    * (
                        (merged["queries_per_decision_base"] - merged["queries_per_decision_pol"])
                        / merged["queries_per_decision_base"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final comparison with tuned ASMF variants")
    parser.add_argument("--variants-file", type=str, default="outputs/tuning/top_variants.json")
    parser.add_argument("--seed-spec", type=str, default="101:120")
    parser.add_argument("--output-dir", type=str, default="outputs/final_comparison")
    parser.add_argument("--scenario-names", type=str, default="")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) - 1, 1))
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--capture-diagnostics", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seed_spec)
    scenarios = _default_scenarios()
    if args.scenario_names.strip():
        wanted = {s.strip() for s in args.scenario_names.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s["name"] in wanted]
    if not scenarios:
        raise ValueError("No scenarios selected")

    variants_path = Path(args.variants_file)
    if not variants_path.exists():
        raise FileNotFoundError(f"Missing variants file: {variants_path}")

    payload = json.loads(variants_path.read_text(encoding="utf-8"))
    top_variants = payload.get("top_variants", [])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[Dict[str, Any], int, str, str, Dict[str, Any], float, bool]] = []

    for scenario in scenarios:
        default_cfg = _engine_config_from_scenario(scenario)
        default_cfg_dict = asdict(default_cfg)
        for seed in seeds:
            tasks.append(
                (
                    scenario,
                    seed,
                    "asmf_default",
                    "asmf",
                    default_cfg_dict,
                    float(args.duration_scale),
                    bool(args.capture_diagnostics),
                )
            )

            for tv in top_variants:
                tasks.append(
                    (
                        scenario,
                        seed,
                        tv["variant_id"],
                        "asmf",
                        tv["config"],
                        float(args.duration_scale),
                        bool(args.capture_diagnostics),
                    )
                )

            for baseline in ["gmsr", "p2c", "least_queue", "random"]:
                tasks.append(
                    (
                        scenario,
                        seed,
                        baseline,
                        baseline,
                        default_cfg_dict,
                        float(args.duration_scale),
                        bool(args.capture_diagnostics),
                    )
                )

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        for row in ex.map(_run_task, tasks, chunksize=8):
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "final_results.csv", index=False)

    summary = (
        df.groupby("policy", as_index=False)
        .agg(
            throughput_mean=("throughput", "mean"),
            throughput_std=("throughput", "std"),
            wait_mean=("avg_wait_time", "mean"),
            backlog_mean=("backlog_area", "mean"),
            qpd_mean=("queries_per_decision", "mean"),
            convergence_time_ms_mean=("convergence_time_ms", "mean"),
            n=("throughput", "size"),
        )
        .sort_values("throughput_mean", ascending=False)
    )
    summary.to_csv(output_dir / "final_summary.csv", index=False)

    pairwise = _pairwise_vs_base(df, base="asmf_default")
    pairwise.to_csv(output_dir / "final_pairwise_vs_default.csv", index=False)

    with open(output_dir / "final_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": seeds,
                "scenario_names": [s["name"] for s in scenarios],
                "variants_file": str(variants_path),
                "top_variants_count": len(top_variants),
            },
            f,
            indent=2,
        )

    print("Final comparison complete.")
    print(f"Rows: {len(df)}")
    print(f"Policies: {sorted(df['policy'].unique().tolist())}")
    print(f"Scenarios: {df['scenario'].nunique()}")
    print(f"Workers: {max(1, int(args.workers))}")
    print(f"Duration scale: {float(args.duration_scale)}")
    print(f"Capture diagnostics: {bool(args.capture_diagnostics)}")
    print(f"Wrote: {output_dir / 'final_results.csv'}")
    print(f"Wrote: {output_dir / 'final_summary.csv'}")
    print(f"Wrote: {output_dir / 'final_pairwise_vs_default.csv'}")
    print(f"Wrote: {output_dir / 'final_config.json'}")


if __name__ == "__main__":
    main()
