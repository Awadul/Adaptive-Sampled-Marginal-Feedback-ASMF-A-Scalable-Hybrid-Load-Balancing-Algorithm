# Proof-by-Simulation Protocol

This project validates the ASMF algorithm via controlled simulation experiments and baseline comparisons.

## Objective

Show that ASMF provides better or comparable throughput and queue stability than classic distributed policies under partial observability.

## Experimental Factors

- Topology: 4 frontends, 12 backends, bipartite constraints.
- Policies compared: `asmf`, `p2c`, `least_queue`, `random`.
- Time model: discrete-time simulation with 100 ms steps.
- State staleness: cache refresh every 300 ms.
- Resource coupling: CPU/MEM/IO/NET pressure increases with queue pressure.
- Trials: multiple random seeds for repeatability.

## Metrics

- Throughput: completed jobs / generated jobs.
- Acceptance rate: routed jobs / generated jobs.
- Rejection rate: rejected jobs / generated jobs.
- Average wait time: queue-time approximation per completed job.
- Backlog area: integral of total queue length over time.
- Max queue observed: worst transient queue pressure.

## Reproducible Procedure

1. Install dependencies.
2. Run `python scripts/run_experiments.py`.
3. Collect artifacts in `outputs/benchmark/`:
   - `benchmark_results.csv`
   - `benchmark_summary.csv`
   - `throughput_boxplot.png`
   - `wait_time_boxplot.png`
   - `backlog_boxplot.png`

## Hypotheses

- H1: ASMF has higher mean throughput than `random` and `least_queue`.
- H2: ASMF has lower mean backlog area than `random`.
- H3: ASMF remains stable under stale score updates via feedback correction.

## Statistical Notes

For publication-level rigor, run at least 30 seeds and compute confidence intervals (bootstrap or t-interval) on throughput and backlog area.
