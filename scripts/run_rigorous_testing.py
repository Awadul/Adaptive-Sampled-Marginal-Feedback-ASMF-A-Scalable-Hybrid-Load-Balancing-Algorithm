from __future__ import annotations

from pathlib import Path

from asmf_lb.experiments import run_rigorous_campaign


def main() -> None:
    output_dir = Path("outputs") / "rigorous"
    df = run_rigorous_campaign(str(output_dir))
    print("Rigorous campaign completed.")
    print(f"Total rows: {len(df)}")
    print(df.groupby("policy")["throughput"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()
