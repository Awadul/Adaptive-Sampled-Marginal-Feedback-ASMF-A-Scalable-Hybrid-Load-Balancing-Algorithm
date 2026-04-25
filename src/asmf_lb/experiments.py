from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .engine import ASMFConfig, ASMFEngine
from .simulator import LoadBalancingSimulator, SimulationConfig, build_default_topology


def run_benchmark_suite(
    output_dir: str,
    seeds: List[int] | None = None,
    config: SimulationConfig | None = None,
    asmf_config: ASMFConfig | None = None,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    seeds = seeds or [11, 17, 23, 31, 47]
    sim_config = config or SimulationConfig()
    algo_config = asmf_config or ASMFConfig()
    policies = ["asmf", "asmf_no_feedback", "asmf_no_sampling", "asmf_no_multiresource", "gmsr", "p2c", "least_queue", "random"]

    rows: List[Dict[str, float]] = []

    for seed in seeds:
        frontends, states = build_default_topology()
        simulator = LoadBalancingSimulator(frontends, states, sim_config, seed=seed)
        for policy in policies:
            engine = ASMFEngine(config=algo_config, seed=seed)
            metrics = simulator.run(policy=policy, engine=engine)
            row = metrics.as_dict()
            row["seed"] = seed
            row["duration_ms"] = sim_config.duration_ms
            row["sample_k"] = algo_config.sample_k
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output / "benchmark_results.csv", index=False)

    summary = (
        df.groupby("policy", as_index=False)
        .agg(
            throughput_mean=("throughput", "mean"),
            acceptance_mean=("acceptance_rate", "mean"),
            rejection_mean=("rejection_rate", "mean"),
            wait_mean=("avg_wait_time", "mean"),
            backlog_mean=("backlog_area", "mean"),
            maxq_mean=("max_queue_observed", "mean"),
        )
        .sort_values("throughput_mean", ascending=False)
    )
    summary.to_csv(output / "benchmark_summary.csv", index=False)

    _plot_metric(df, "throughput", output / "throughput_boxplot.png", "Throughput by Policy")
    _plot_metric(df, "avg_wait_time", output / "wait_time_boxplot.png", "Average Wait Time by Policy")
    _plot_metric(df, "backlog_area", output / "backlog_boxplot.png", "Backlog Area by Policy")

    return df


def _plot_metric(df: pd.DataFrame, metric: str, outpath: Path, title: str) -> None:
    # Determine available policies in the dataframe
    available = df["policy"].unique().tolist()
    # Preferred order: asmf, gmsr, p2c, least_queue, random
    preferred_order = [
        "asmf",
        "asmf_no_feedback",
        "asmf_no_sampling",
        "asmf_no_multiresource",
        "gmsr",
        "p2c",
        "least_queue",
        "random",
    ]
    order = [p for p in preferred_order if p in available]
    # Add any remaining policies not in preferred order
    for p in available:
        if p not in order:
            order.append(p)
    
    data = [df[df["policy"] == p][metric] for p in order]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=order, patch_artist=True)
    plt.title(title)
    plt.ylabel(metric)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def experiment_config_dump(output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sim_cfg = SimulationConfig()
    asmf_cfg = ASMFConfig()

    with open(output / "config_used.json", "w", encoding="utf-8") as f:
        import json

        json.dump(
            {
                "simulation": asdict(sim_cfg),
                "asmf": asdict(asmf_cfg),
                "policies": [
                    "asmf",
                    "asmf_no_feedback",
                    "asmf_no_sampling",
                    "asmf_no_multiresource",
                    "gmsr",
                    "p2c",
                    "least_queue",
                    "random",
                ],
            },
            f,
            indent=2,
        )


def run_rigorous_campaign(
    output_dir: str,
    seeds: List[int] | None = None,
    scenarios: List[Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    seeds = seeds or list(range(101, 121))
    scenarios = scenarios or _default_scenarios()
    policies = [
        "asmf",
        "asmf_no_feedback",
        "asmf_no_sampling",
        "asmf_no_multiresource",
        "gmsr",
        "p2c",
        "least_queue",
        "random",
    ]

    rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        sim_cfg = SimulationConfig(
            duration_ms=int(scenario["duration_ms"]),
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
        )
        asmf_cfg = ASMFConfig(
            sample_k=int(scenario["sample_k"]),
            resource_weights=tuple(scenario["resource_weights"]),
            eta=float(scenario["eta"]),
            threshold=float(scenario["threshold"]),
            min_correction=float(scenario.get("min_correction", 0.3)),
            max_correction=float(scenario.get("max_correction", 2.5)),
        )

        for seed in seeds:
            frontends, states = build_default_topology(
                num_frontends=int(scenario["num_frontends"]),
                num_backends=int(scenario["num_backends"]),
            )
            simulator = LoadBalancingSimulator(frontends, states, sim_cfg, seed=seed)
            for policy in policies:
                engine = ASMFEngine(config=asmf_cfg, seed=seed)
                metrics = simulator.run(policy=policy, engine=engine)
                row = metrics.as_dict()
                row["seed"] = seed
                row["scenario"] = scenario["name"]
                row["duration_ms"] = sim_cfg.duration_ms
                row["time_step_ms"] = sim_cfg.time_step_ms
                row["arrivals_per_step"] = sim_cfg.arrivals_per_step
                row["stale_update_interval_ms"] = sim_cfg.stale_update_interval_ms
                row["workload_type"] = sim_cfg.workload_type
                row["num_frontends"] = scenario["num_frontends"]
                row["num_backends"] = scenario["num_backends"]
                row["sample_k"] = asmf_cfg.sample_k
                row["threshold"] = asmf_cfg.threshold
                row["eta"] = asmf_cfg.eta
                row["failure_fraction"] = sim_cfg.failure_fraction
                row["load_spike_multiplier"] = sim_cfg.load_spike_multiplier
                row["degrade_factor"] = sim_cfg.degrade_factor
                rows.append(row)

                for rec in metrics.trace_records:
                    trace_rows.append(
                        {
                            "scenario": scenario["name"],
                            "seed": seed,
                            "policy": policy,
                            "time_ms": int(rec["time_ms"]),
                            "queue_mean": rec["queue_mean"],
                            "queue_var": rec["queue_var"],
                            "rejection_rate": rec["rejection_rate"],
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(output / "rigorous_results.csv", index=False)
    if trace_rows:
        pd.DataFrame(trace_rows).to_csv(output / "rigorous_time_series.csv", index=False)

    summary = _summarize_with_ci(df)
    summary.to_csv(output / "rigorous_summary_ci.csv", index=False)

    comparisons = _pairwise_improvement(df)
    comparisons.to_csv(output / "asmf_pairwise_improvements.csv", index=False)

    _plot_metric(df, "throughput", output / "rigorous_throughput_boxplot.png", "Rigorous Throughput by Policy")
    _plot_metric(df, "avg_wait_time", output / "rigorous_wait_boxplot.png", "Rigorous Wait Time by Policy")
    _plot_metric(df, "backlog_area", output / "rigorous_backlog_boxplot.png", "Rigorous Backlog by Policy")
    _plot_metric(df, "state_queries", output / "rigorous_state_queries_boxplot.png", "Rigorous State Query Cost by Policy")
    _plot_metric(
        df,
        "bytes_transferred_est",
        output / "rigorous_bytes_transferred_boxplot.png",
        "Rigorous Communication Bytes by Policy",
    )
    _plot_metric(df, "convergence_time_ms", output / "rigorous_convergence_time_boxplot.png", "Rigorous Convergence Time by Policy")

    report = _build_markdown_report(df, summary, comparisons)
    with open(output / "rigorous_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    with open(output / "rigorous_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": seeds,
                "scenario_count": len(scenarios),
                "scenarios": scenarios,
                "policies": policies,
            },
            f,
            indent=2,
        )

    return df


def _summarize_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metrics = [
        "throughput",
        "avg_wait_time",
        "backlog_area",
        "max_queue_observed",
        "state_queries",
        "bytes_transferred_est",
        "queries_per_decision",
        "convergence_time_ms",
        "mean_queue_variance",
        "oscillation_index",
    ]

    grouped = df.groupby("policy")
    for policy, g in grouped:
        n = len(g)
        row: Dict[str, Any] = {"policy": policy, "n": n}
        for m in metrics:
            mean = float(g[m].mean())
            std = float(g[m].std(ddof=1)) if n > 1 else 0.0
            half_ci = 1.96 * std / max(np.sqrt(n), 1e-9)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95_low"] = mean - half_ci
            row[f"{m}_ci95_high"] = mean + half_ci
        rows.append(row)

    return pd.DataFrame(rows).sort_values("throughput_mean", ascending=False)


def _pairwise_improvement(df: pd.DataFrame) -> pd.DataFrame:
    baselines = [p for p in sorted(df["policy"].unique().tolist()) if p != "asmf"]
    rows: List[Dict[str, Any]] = []

    for base in baselines:
        merged = _join_asmf_with_baseline(df, base)
        rows.append(
            {
                "baseline": base,
                "throughput_gain_pct": _pct((merged["throughput_asmf"] - merged["throughput_base"]) / merged["throughput_base"]),
                "wait_reduction_pct": _pct((merged["avg_wait_time_base"] - merged["avg_wait_time_asmf"]) / merged["avg_wait_time_base"]),
                "backlog_reduction_pct": _pct((merged["backlog_area_base"] - merged["backlog_area_asmf"]) / merged["backlog_area_base"]),
                "max_queue_reduction_pct": _pct((merged["max_queue_observed_base"] - merged["max_queue_observed_asmf"]) / merged["max_queue_observed_base"]),
                "asmf_wins_throughput_pct": _pct((merged["throughput_asmf"] > merged["throughput_base"]).astype(float)),
                "asmf_wins_wait_pct": _pct((merged["avg_wait_time_asmf"] < merged["avg_wait_time_base"]).astype(float)),
                "asmf_wins_backlog_pct": _pct((merged["backlog_area_asmf"] < merged["backlog_area_base"]).astype(float)),
            }
        )

    return pd.DataFrame(rows)


def _join_asmf_with_baseline(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    id_cols = ["scenario", "seed"]
    metrics = ["throughput", "avg_wait_time", "backlog_area", "max_queue_observed"]

    a = df[df["policy"] == "asmf"][id_cols + metrics].copy()
    b = df[df["policy"] == baseline][id_cols + metrics].copy()
    a.columns = id_cols + [f"{m}_asmf" for m in metrics]
    b.columns = id_cols + [f"{m}_base" for m in metrics]
    return a.merge(b, on=id_cols, how="inner")


def _pct(series: pd.Series) -> float:
    return float(100.0 * series.mean())


def _build_markdown_report(df: pd.DataFrame, summary: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    scenario_counts = (
        df.groupby(["scenario", "policy"]).size().reset_index(name="runs").sort_values(["scenario", "policy"])
    )
    workload_counts = (
        df.groupby(["workload_type", "policy"]).size().reset_index(name="runs").sort_values(["workload_type", "policy"])
    )
    comm_summary = (
        df.groupby("policy", as_index=False)
        .agg(
            state_queries_mean=("state_queries", "mean"),
            bytes_transferred_est_mean=("bytes_transferred_est", "mean"),
            queries_per_decision_mean=("queries_per_decision", "mean"),
        )
        .sort_values("queries_per_decision_mean")
    )
    convergence_summary = (
        df.groupby("policy", as_index=False)
        .agg(
            convergence_time_ms_mean=("convergence_time_ms", "mean"),
            mean_queue_variance_mean=("mean_queue_variance", "mean"),
            oscillation_index_mean=("oscillation_index", "mean"),
        )
        .sort_values("convergence_time_ms_mean")
    )

    lines = [
        "# Rigorous Benchmark Report",
        "",
        "## Setup",
        "",
        f"- Total runs: {len(df)}",
        f"- Unique scenarios: {df['scenario'].nunique()}",
        f"- Seeds per scenario: {df['seed'].nunique()}",
        "- Policies: asmf, asmf_no_feedback, asmf_no_sampling, asmf_no_multiresource, gmsr, p2c, least_queue, random",
        "",
        "## Policy Summary With 95% CI",
        "",
        summary.to_markdown(index=False),
        "",
        "## ASMF Pairwise Improvements",
        "",
        comparisons.to_markdown(index=False),
        "",
        "## Communication Cost Summary",
        "",
        comm_summary.to_markdown(index=False),
        "",
        "## Convergence Summary",
        "",
        convergence_summary.to_markdown(index=False),
        "",
        "## Workload Coverage",
        "",
        workload_counts.to_markdown(index=False),
        "",
        "## Scenario Coverage",
        "",
        scenario_counts.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def _default_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "name": "poisson_small",
            "duration_ms": 600_000,
            "arrivals_per_step": 6.0,
            "stale_update_interval_ms": 300,
            "num_frontends": 4,
            "num_backends": 10,
            "sample_k": 2,
            "threshold": 0.12,
            "eta": 0.05,
            "resource_weights": [0.4, 0.25, 0.2, 0.15],
            "workload_type": "poisson",
        },
        {
            "name": "poisson_medium",
            "duration_ms": 600_000,
            "arrivals_per_step": 8.0,
            "stale_update_interval_ms": 300,
            "num_frontends": 8,
            "num_backends": 50,
            "sample_k": 3,
            "threshold": 0.10,
            "eta": 0.06,
            "resource_weights": [0.4, 0.25, 0.2, 0.15],
            "workload_type": "poisson",
        },
        {
            "name": "poisson_large",
            "duration_ms": 600_000,
            "arrivals_per_step": 10.0,
            "stale_update_interval_ms": 400,
            "num_frontends": 20,
            "num_backends": 220,
            "sample_k": 4,
            "threshold": 0.10,
            "eta": 0.05,
            "resource_weights": [0.35, 0.25, 0.2, 0.2],
            "workload_type": "poisson",
        },
        {
            "name": "bursty_medium",
            "duration_ms": 600_000,
            "arrivals_per_step": 8.0,
            "stale_update_interval_ms": 300,
            "num_frontends": 8,
            "num_backends": 50,
            "sample_k": 3,
            "threshold": 0.10,
            "eta": 0.05,
            "resource_weights": [0.4, 0.25, 0.2, 0.15],
            "workload_type": "bursty",
            "burst_on_multiplier": 3.0,
            "burst_off_multiplier": 0.3,
            "burst_period_steps": 45,
        },
        {
            "name": "zipf_skew_medium",
            "duration_ms": 600_000,
            "arrivals_per_step": 8.0,
            "stale_update_interval_ms": 300,
            "num_frontends": 8,
            "num_backends": 50,
            "sample_k": 3,
            "threshold": 0.11,
            "eta": 0.05,
            "resource_weights": [0.2, 0.45, 0.2, 0.15],
            "workload_type": "zipf_skew",
            "zipf_alpha": 1.35,
        },
        {
            "name": "failure_medium",
            "duration_ms": 600_000,
            "arrivals_per_step": 8.0,
            "stale_update_interval_ms": 300,
            "num_frontends": 8,
            "num_backends": 50,
            "sample_k": 3,
            "threshold": 0.10,
            "eta": 0.05,
            "resource_weights": [0.4, 0.25, 0.2, 0.15],
            "workload_type": "poisson",
            "backend_failure_at_ms": 240_000,
            "failure_fraction": 0.2,
        },
        {
            "name": "spike_medium",
            "duration_ms": 600_000,
            "arrivals_per_step": 8.0,
            "stale_update_interval_ms": 300,
            "num_frontends": 8,
            "num_backends": 50,
            "sample_k": 3,
            "threshold": 0.10,
            "eta": 0.05,
            "resource_weights": [0.4, 0.25, 0.2, 0.15],
            "workload_type": "poisson",
            "load_spike_at_ms": 300_000,
            "load_spike_multiplier": 2.0,
            "load_spike_duration_ms": 60_000,
        },
        {
            "name": "degraded_medium",
            "duration_ms": 600_000,
            "arrivals_per_step": 8.0,
            "stale_update_interval_ms": 300,
            "num_frontends": 8,
            "num_backends": 50,
            "sample_k": 3,
            "threshold": 0.10,
            "eta": 0.05,
            "resource_weights": [0.4, 0.25, 0.2, 0.15],
            "workload_type": "poisson",
            "degrade_at_ms": 240_000,
            "degrade_factor": 0.7,
        },
    ]
