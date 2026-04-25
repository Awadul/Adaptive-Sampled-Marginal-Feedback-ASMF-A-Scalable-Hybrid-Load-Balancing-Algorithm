# Proof-by-Simulation Protocol

This project validates the ASMF algorithm via controlled simulation experiments and baseline comparisons.

## Objective

Show that ASMF provides better or comparable throughput and queue stability than classic distributed policies under partial observability.

## Experimental Factors

- Topology tiers: small (10 backends), medium (50), large (220) with bipartite constraints.
- Policies compared: `asmf`, `asmf_no_feedback`, `asmf_no_sampling`, `asmf_no_multiresource`, `gmsr`, `p2c`, `least_queue`, `random`.
- Workload families: `poisson`, `bursty` (ON/OFF), `zipf_skew`.
- Time model: discrete-time simulation with 100 ms steps.
- State staleness: cache refresh every 300 ms.
- Resource coupling: CPU/MEM/IO/NET pressure increases with queue pressure.
- Stress events: backend failure, load spike, service degradation.
- Trials: 20 random seeds per scenario for repeatability.

## Metrics

- Throughput: completed jobs / generated jobs.
- Acceptance rate: routed jobs / generated jobs.
- Rejection rate: rejected jobs / generated jobs.
- Average wait time: queue-time approximation per completed job.
- Backlog area: integral of total queue length over time.
- Max queue observed: worst transient queue pressure.
- State update count: number of cache update broadcasts.
- Query count: backend-state lookups performed by routing.
- Estimated bytes transferred: communication-volume proxy.
- Queries per decision: communication intensity per routed request.
- Convergence time: first stable window under tolerance.
- Oscillation index and mean queue variance over time.

## Reproducible Procedure

1. Install dependencies.
2. Run `python scripts/run_experiments.py`.
3. Run `python scripts/run_rigorous_testing.py`.
4. Run `python scripts/analyze_rigorous_results.py`.
5. Collect artifacts in `outputs/benchmark/`:
   - `benchmark_results.csv`
   - `benchmark_summary.csv`
   - `throughput_boxplot.png`
   - `wait_time_boxplot.png`
   - `backlog_boxplot.png`
6. Collect rigorous artifacts in `outputs/rigorous/`:
   - `rigorous_results.csv`
   - `rigorous_summary_ci.csv`
   - `asmf_pairwise_improvements.csv`
   - `scenario_pairwise_improvements.csv`
   - `pairwise_stat_tests.csv`
   - `rigorous_time_series.csv`

## Hypotheses

- H1: ASMF maintains throughput parity with strong baselines under all workloads.
- H2: ASMF reduces wait/backlog versus baselines under bursty and skewed load.
- H3: ASMF communication overhead is lower than global-state methods.
- H4: Feedback, sampling, and multi-resource scoring each contribute measurable gains (ablation).
- H5: ASMF converges faster and oscillates less under stress events.

## Statistical Notes

For publication-level rigor, compute:

- 95% CI per policy metric
- bootstrap paired CI for ASMF vs each baseline
- paired t-test and Wilcoxon signed-rank tests per comparison metric
