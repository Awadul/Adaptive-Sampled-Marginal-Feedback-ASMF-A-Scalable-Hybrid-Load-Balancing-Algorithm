# Adaptive Sampled Marginal Feedback (ASMF) Load Balancer

ASMF is a hybrid distributed load-balancing algorithm designed for scalable and adaptive routing under partial system information.

It combines:

- GMSR-style marginal scoring
- power-of-k sampling
- DFLB-inspired feedback correction
- multi-resource load awareness
- low-overhead state sharing

## Core Idea

For backend $j$, ASMF computes:

$$
L_j = \alpha CPU_j + \beta MEM_j + \gamma IO_j + \delta NET_j
$$

$$
S_j = \frac{\mu_j}{(1 + q_j)(1 + L_j)}
$$

Feedback update:

$$
\Delta_j = \lambda^{in}_j - \lambda^{svc}_j
$$

$$
C_j(t+1) = \operatorname{clip}\left(C_j(t) - \eta \Delta_j, C_{min}, C_{max}\right)
$$

Final score used for routing:

$$
S'_j = S_j \cdot C_j
$$

Routing rule:

1. sample $k$ allowed backends
2. choose backend with maximum cached $S'_j$
3. reject/delay if $\max S'_j < \tau$

## Repository Layout

- `src/asmf_lb/engine.py`: ASMF policy and baselines
- `src/asmf_lb/simulator.py`: discrete-event simulation and topology generation
- `src/asmf_lb/experiments.py`: benchmark suite, summary generation, and plots
- `scripts/run_experiments.py`: full benchmark run
- `scripts/quick_demo.py`: short local demonstration
- `tests/test_engine.py`: routing and feedback unit tests
- `docs/PROOF_SIMULATIONS.md`: simulation proof protocol
- `docs/GITHUB_PUBLISH_PROTOCOL.md`: GitHub submission workflow

## Quick Start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run Demo

```bash
python scripts/quick_demo.py
```

### 3. Run Full Benchmarks

```bash
python scripts/run_experiments.py
```

### 4. Run Rigorous Multi-Scenario Testing

```bash
python scripts/run_rigorous_testing.py
```

This runs a longer campaign (5 scenarios x 20 seeds) with 10-minute simulated duration per scenario and writes artifacts to `outputs/rigorous/`.

Outputs are generated in `outputs/benchmark/`:

- `benchmark_results.csv`
- `benchmark_summary.csv`
- `throughput_boxplot.png`
- `wait_time_boxplot.png`
- `backlog_boxplot.png`
- `config_used.json`

### 5. Run Tests

```bash
pytest -q
```

## Policies Included

- `asmf`: proposed algorithm
- `p2c`: power-of-k by shortest sampled queue
- `least_queue`: global least queue in allowed set
- `random`: random backend in allowed set

## Academic Claims Supported

- communication complexity is $O(k)$ per request
- feedback correction reduces sustained overload drift
- multi-resource scoring avoids queue-only bias
- sampled routing provides near-optimal behavior at lower overhead

## Suggested Submission Package

1. Source code and tests
2. Benchmark outputs from `outputs/benchmark/`
3. `docs/PROOF_SIMULATIONS.md`
4. `docs/GITHUB_PUBLISH_PROTOCOL.md`
5. Short report section citing ASMF equations and results

## License

MIT
