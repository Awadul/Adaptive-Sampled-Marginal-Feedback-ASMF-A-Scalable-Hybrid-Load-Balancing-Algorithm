from __future__ import annotations

import argparse
import os
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def _random_weight_vector(rng: np.random.Generator) -> tuple[float, float, float, float]:
    cpu = float(rng.uniform(0.45, 0.65))
    mem = float(rng.uniform(0.10, 0.25))
    io = float(rng.uniform(0.05, 0.20))
    net = float(rng.uniform(0.05, 0.20))
    total = cpu + mem + io + net
    return (cpu / total, mem / total, io / total, net / total)


def _sample_config(rng: np.random.Generator) -> ASMFConfig:
    weights = _random_weight_vector(rng)
    return ASMFConfig(
        sample_k=int(rng.integers(2, 5)),
        resource_weights=weights,
        eta=float(rng.uniform(0.003, 0.02)),
        threshold=float(rng.uniform(0.08, 0.14)),
        min_correction=float(rng.uniform(0.70, 0.90)),
        max_correction=float(rng.uniform(1.30, 1.80)),
    )


def _evaluate_single_run(
    scenario: Dict[str, Any],
    seed: int,
    config: ASMFConfig,
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
    engine = ASMFEngine(config=config, seed=seed)
    metrics = simulator.run(policy="asmf", engine=engine).as_dict()
    metrics["seed"] = seed
    metrics["scenario"] = scenario["name"]
    metrics["workload_type"] = sim_cfg.workload_type
    metrics["num_frontends"] = scenario["num_frontends"]
    metrics["num_backends"] = scenario["num_backends"]
    return metrics


def _evaluate_task(task: Tuple[Dict[str, Any], int, Dict[str, Any], str, float, bool]) -> Dict[str, Any]:
    scenario, seed, cfg_dict, cfg_id, duration_scale, capture_diagnostics = task
    cfg = ASMFConfig(**cfg_dict)
    row = _evaluate_single_run(
        scenario,
        seed,
        cfg,
        duration_scale=duration_scale,
        capture_diagnostics=capture_diagnostics,
    )
    row["config_id"] = cfg_id
    row.update({f"cfg_{k}": v for k, v in asdict(cfg).items()})
    return row


def _aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        rows.groupby("config_id", as_index=False)
        .agg(
            throughput_mean=("throughput", "mean"),
            wait_mean=("avg_wait_time", "mean"),
            backlog_mean=("backlog_area", "mean"),
            maxq_mean=("max_queue_observed", "mean"),
            qpd_mean=("queries_per_decision", "mean"),
            bytes_mean=("bytes_transferred_est", "mean"),
            convergence_mean=("convergence_time_ms", "mean"),
            throughput_std=("throughput", "std"),
            wait_std=("avg_wait_time", "std"),
            runs=("throughput", "size"),
        )
    )

    # Normalize with min-max to combine objectives safely.
    def _norm(series: pd.Series, higher_is_better: bool) -> pd.Series:
        min_v = float(series.min())
        max_v = float(series.max())
        if abs(max_v - min_v) < 1e-12:
            return pd.Series(np.full(len(series), 0.5), index=series.index)
        scaled = (series - min_v) / (max_v - min_v)
        return scaled if higher_is_better else (1.0 - scaled)

    grouped["throughput_score"] = _norm(grouped["throughput_mean"], True)
    grouped["wait_score"] = _norm(grouped["wait_mean"], False)
    grouped["backlog_score"] = _norm(grouped["backlog_mean"], False)
    grouped["comm_score"] = _norm(grouped["qpd_mean"], False)

    # Weighted objective aligned with tradeoff claim.
    grouped["composite_score"] = (
        0.40 * grouped["throughput_score"]
        + 0.30 * grouped["wait_score"]
        + 0.20 * grouped["backlog_score"]
        + 0.10 * grouped["comm_score"]
    )
    return grouped.sort_values("composite_score", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted ASMF parameter tuning search")
    parser.add_argument("--n-configs", type=int, default=180, help="Number of random configs to evaluate")
    parser.add_argument("--seed-spec", type=str, default="101:110", help="Seeds as start:end or comma list")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top variants to export")
    parser.add_argument("--scenario-names", type=str, default="", help="Comma-separated scenario names to include")
    parser.add_argument("--rng-seed", type=int, default=2026, help="RNG seed for config sampling")
    parser.add_argument("--output-dir", type=str, default="outputs/tuning", help="Output directory")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) - 1, 1), help="Parallel worker processes")
    parser.add_argument("--duration-scale", type=float, default=0.5, help="Scale scenario duration for faster tuning")
    parser.add_argument("--capture-diagnostics", action="store_true", help="Keep expensive per-step traces/snapshots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seed_spec)
    scenarios = _default_scenarios()
    if args.scenario_names.strip():
        wanted = {s.strip() for s in args.scenario_names.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s["name"] in wanted]

    if not scenarios:
        raise ValueError("No scenarios selected for tuning")

    rng = np.random.default_rng(args.rng_seed)
    config_bank: List[tuple[str, ASMFConfig]] = [("asmf_default", _engine_config_from_scenario(scenarios[0]))]
    for i in range(args.n_configs):
        config_bank.append((f"cand_{i+1:03d}", _sample_config(rng)))

    tasks: List[Tuple[Dict[str, Any], int, Dict[str, Any], str, float, bool]] = []
    for cfg_id, cfg in config_bank:
        cfg_dict = asdict(cfg)
        for scenario in scenarios:
            for seed in seeds:
                tasks.append(
                    (
                        scenario,
                        seed,
                        cfg_dict,
                        cfg_id,
                        float(args.duration_scale),
                        bool(args.capture_diagnostics),
                    )
                )

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        for row in ex.map(_evaluate_task, tasks, chunksize=8):
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "asmf_tuning_raw.csv", index=False)

    ranking = _aggregate(df)

    cfg_df = (
        df.groupby("config_id", as_index=False)
        .agg(
            cfg_sample_k=("cfg_sample_k", "first"),
            cfg_resource_weights=("cfg_resource_weights", "first"),
            cfg_eta=("cfg_eta", "first"),
            cfg_threshold=("cfg_threshold", "first"),
            cfg_min_correction=("cfg_min_correction", "first"),
            cfg_max_correction=("cfg_max_correction", "first"),
        )
    )
    ranking = ranking.merge(cfg_df, on="config_id", how="left")
    ranking.to_csv(output_dir / "asmf_tuning_ranked.csv", index=False)

    top = ranking.head(args.top_k).copy()
    variant_payload = []
    for idx, r in enumerate(top.itertuples(index=False), start=1):
        variant_payload.append(
            {
                "variant_id": f"asmf_tuned_v{idx}",
                "source_config_id": r.config_id,
                "composite_score": float(r.composite_score),
                "config": {
                    "sample_k": int(r.cfg_sample_k),
                    "resource_weights": list(r.cfg_resource_weights),
                    "eta": float(r.cfg_eta),
                    "threshold": float(r.cfg_threshold),
                    "min_correction": float(r.cfg_min_correction),
                    "max_correction": float(r.cfg_max_correction),
                },
            }
        )

    with open(output_dir / "top_variants.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": seeds,
                "scenario_names": [s["name"] for s in scenarios],
                "n_configs_tested": len(config_bank),
                "top_variants": variant_payload,
            },
            f,
            indent=2,
        )

    print("Tuning complete.")
    print(f"Evaluated configs: {len(config_bank)}")
    print(f"Scenarios: {len(scenarios)}, Seeds: {len(seeds)}")
    print(f"Workers: {max(1, int(args.workers))}")
    print(f"Duration scale: {float(args.duration_scale)}")
    print(f"Capture diagnostics: {bool(args.capture_diagnostics)}")
    print(f"Raw rows: {len(df)}")
    print(f"Wrote: {output_dir / 'asmf_tuning_raw.csv'}")
    print(f"Wrote: {output_dir / 'asmf_tuning_ranked.csv'}")
    print(f"Wrote: {output_dir / 'top_variants.json'}")


if __name__ == "__main__":
    main()
