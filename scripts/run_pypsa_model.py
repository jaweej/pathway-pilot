from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pathway_pilot.config import DEV_DATA_DIR
from pathway_pilot.model_config import load_config
from pathway_pilot.model_inputs import build_model_inputs
from pathway_pilot.outputs import write_model_outputs
from pathway_pilot.solve import solve_model


def main() -> None:
    run_started_at = datetime.now().astimezone()
    solve_start = perf_counter()
    cfg = load_config(Path("config/model_config.yaml"))
    tvar_dir = DEV_DATA_DIR / "pathway-pilot" / "tvar"
    capacity_factors = pd.read_parquet(tvar_dir / "pecd_capacity_factors_2030_2040.parquet")
    demand = pd.read_parquet(tvar_dir / "tyndp_demand_profiles_2030_2040.parquet")

    data = build_model_inputs(
        capacity_factors=capacity_factors,
        demand=demand,
        periods=cfg.investment_periods,
        weather_year=1982,
    )
    network, status, condition = solve_model(cfg, data)
    run_time_seconds = perf_counter() - solve_start
    if (status, condition) != ("ok", "optimal"):
        raise RuntimeError(f"PyPSA solve failed: status={status}, condition={condition}")

    output_dir = DEV_DATA_DIR / "pathway-pilot" / "output"
    write_model_outputs(network, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "solve_times.txt").open("a", encoding="utf-8") as handle:
        handle.write(
            f"run_at: {run_started_at.isoformat()}, "
            f"run_time_secs: {run_time_seconds:.3f}\n"
        )
    print(f"Wrote model outputs to {output_dir}")
    print(f"Solve run time: {run_time_seconds:.3f} seconds")


if __name__ == "__main__":
    main()
