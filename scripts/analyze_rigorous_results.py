from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _ci95_mean(series: pd.Series) -> tuple[float, float, float, float]:
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / max(np.sqrt(n), 1e-9)
    return mean, std, mean - half, mean + half


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 123) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = np.mean(sample)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _paired_t_test(vals: np.ndarray) -> tuple[float, float]:
    n = vals.size
    if n < 2:
        return 0.0, 1.0
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    if std <= 1e-12:
        return 0.0, 1.0
    t_stat = mean / (std / math.sqrt(n))
    # Normal approximation is adequate for n=100 paired samples.
    p = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return t_stat, float(max(min(p, 1.0), 0.0))


def _wilcoxon_signed_rank(vals: np.ndarray) -> tuple[float, float]:
    nonzero = vals[np.abs(vals) > 1e-12]
    n = nonzero.size
    if n == 0:
        return 0.0, 1.0

    abs_vals = np.abs(nonzero)
    order = np.argsort(abs_vals)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(abs_vals[order[j]] - abs_vals[order[i]]) <= 1e-12:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    w_plus = float(np.sum(ranks[nonzero > 0]))
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 1e-12:
        return w_plus, 1.0
    z = (w_plus - mean_w) / math.sqrt(var_w)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return w_plus, float(max(min(p, 1.0), 0.0))


def _pairwise(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    id_cols = ["scenario", "seed"]
    metrics = ["throughput", "avg_wait_time", "backlog_area", "max_queue_observed"]

    a = df[df["policy"] == "asmf"][id_cols + metrics].copy()
    b = df[df["policy"] == baseline][id_cols + metrics].copy()
    a.columns = id_cols + [f"{m}_asmf" for m in metrics]
    b.columns = id_cols + [f"{m}_base" for m in metrics]
    merged = a.merge(b, on=id_cols, how="inner")
    if merged.empty:
        return merged

    merged["baseline"] = baseline
    merged["throughput_gain_pct"] = (merged["throughput_asmf"] - merged["throughput_base"]) / merged["throughput_base"] * 100.0
    merged["wait_reduction_pct"] = (merged["avg_wait_time_base"] - merged["avg_wait_time_asmf"]) / merged["avg_wait_time_base"] * 100.0
    merged["backlog_reduction_pct"] = (merged["backlog_area_base"] - merged["backlog_area_asmf"]) / merged["backlog_area_base"] * 100.0
    merged["max_queue_reduction_pct"] = (
        (merged["max_queue_observed_base"] - merged["max_queue_observed_asmf"]) / merged["max_queue_observed_base"] * 100.0
    )
    return merged


def main() -> None:
    out_dir = Path("outputs") / "rigorous"
    src = out_dir / "rigorous_results.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}. Run scripts/run_rigorous_testing.py first.")

    df = pd.read_csv(src)

    # Policy-level summary with simple CI.
    rows: List[Dict[str, float]] = []
    for policy, g in df.groupby("policy"):
        row: Dict[str, float] = {"policy": policy, "n": len(g)}
        for m in ["throughput", "avg_wait_time", "backlog_area", "max_queue_observed"]:
            mean, std, lo, hi = _ci95_mean(g[m])
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95_low"] = lo
            row[f"{m}_ci95_high"] = hi
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("throughput_mean", ascending=False)
    summary.to_csv(out_dir / "rigorous_summary_ci.csv", index=False)

    # Scenario-level mean table.
    metrics = ["throughput", "avg_wait_time", "backlog_area", "max_queue_observed"]
    scenario_means = df.groupby(["scenario", "policy"], as_index=False)[metrics].mean()
    scenario_means.to_csv(out_dir / "scenario_policy_means.csv", index=False)

    # Pairwise detailed and scenario aggregate tables.
    baselines = [p for p in sorted(df["policy"].unique().tolist()) if p != "asmf"]
    detailed_parts = []
    scenario_rows = []
    overall_rows = []
    test_rows = []

    for base in baselines:
        merged = _pairwise(df, base)
        if merged.empty:
            continue
        detailed_parts.append(merged)

        # Overall paired stats + bootstrap CI.
        entry: Dict[str, float | str] = {"baseline": base, "n": len(merged)}
        for metric in ["throughput_gain_pct", "wait_reduction_pct", "backlog_reduction_pct", "max_queue_reduction_pct"]:
            vals = merged[metric].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            lo, hi = _bootstrap_ci(vals)
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_boot95_low"] = lo
            entry[f"{metric}_boot95_high"] = hi
            entry[f"{metric}_ci_excludes_zero"] = bool((lo > 0.0) or (hi < 0.0))

            t_stat, t_p = _paired_t_test(vals)
            w_stat, w_p = _wilcoxon_signed_rank(vals)
            test_rows.append(
                {
                    "baseline": base,
                    "metric": metric,
                    "n": len(vals),
                    "paired_t_stat": t_stat,
                    "paired_t_pvalue": t_p,
                    "wilcoxon_w": w_stat,
                    "wilcoxon_pvalue": w_p,
                }
            )

        entry["asmf_wins_throughput_pct"] = float((merged["throughput_asmf"] > merged["throughput_base"]).mean() * 100.0)
        entry["asmf_wins_wait_pct"] = float((merged["avg_wait_time_asmf"] < merged["avg_wait_time_base"]).mean() * 100.0)
        entry["asmf_wins_backlog_pct"] = float((merged["backlog_area_asmf"] < merged["backlog_area_base"]).mean() * 100.0)
        overall_rows.append(entry)

        # Per-scenario paired improvements.
        for scenario, sg in merged.groupby("scenario"):
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "baseline": base,
                    "n": len(sg),
                    "throughput_gain_pct": float(sg["throughput_gain_pct"].mean()),
                    "wait_reduction_pct": float(sg["wait_reduction_pct"].mean()),
                    "backlog_reduction_pct": float(sg["backlog_reduction_pct"].mean()),
                    "max_queue_reduction_pct": float(sg["max_queue_reduction_pct"].mean()),
                }
            )

    if detailed_parts:
        detailed = pd.concat(detailed_parts, ignore_index=True)
        detailed.to_csv(out_dir / "pairwise_detailed.csv", index=False)

    pd.DataFrame(overall_rows).to_csv(out_dir / "asmf_pairwise_improvements.csv", index=False)
    pd.DataFrame(scenario_rows).sort_values(["scenario", "baseline"]).to_csv(
        out_dir / "scenario_pairwise_improvements.csv", index=False
    )
    pd.DataFrame(test_rows).to_csv(out_dir / "pairwise_stat_tests.csv", index=False)

    print("Rigorous analysis complete.")
    print("Wrote:")
    for name in [
        "rigorous_summary_ci.csv",
        "scenario_policy_means.csv",
        "pairwise_detailed.csv",
        "asmf_pairwise_improvements.csv",
        "scenario_pairwise_improvements.csv",
        "pairwise_stat_tests.csv",
    ]:
        print(f"- outputs/rigorous/{name}")


if __name__ == "__main__":
    main()
