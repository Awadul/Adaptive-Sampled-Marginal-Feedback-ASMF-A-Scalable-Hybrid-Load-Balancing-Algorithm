from __future__ import annotations

from pathlib import Path

from asmf_lb.experiments import experiment_config_dump, run_benchmark_suite


def main() -> None:
    output_dir = Path("outputs") / "benchmark"
    experiment_config_dump(str(output_dir))
    df = run_benchmark_suite(str(output_dir))
    print("Experiment completed.")
    print(df.groupby("policy")["throughput"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()
