from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pathway_pilot.config import DEV_DATA_DIR
from pathway_pilot.model_config import ModelConfig, load_config, with_active_model
from pathway_pilot.model_inputs import build_model_inputs
from pathway_pilot.outputs import write_model_outputs
from pathway_pilot.solve import solve_model


STAGING_ROOT = Path(".tmp") / "model_outputs"


def scenario_output_dir(
    output_root: Path,
    model_case: str,
    weather_year: int,
    climate_year_count: int,
) -> Path:
    if climate_year_count == 1:
        return output_root / model_case
    return output_root / model_case / f"weather_{weather_year}"


def load_time_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    tvar_dir = DEV_DATA_DIR / "pathway-pilot" / "tvar"
    capacity_factors = pd.read_parquet(tvar_dir / "pecd_capacity_factors_2030_2040.parquet")
    demand = pd.read_parquet(tvar_dir / "tyndp_demand_profiles_2030_2040.parquet")
    return capacity_factors, demand


def run_model_case(
    cfg: ModelConfig,
    capacity_factors: pd.DataFrame,
    demand: pd.DataFrame,
    output_dir: Path,
    weather_year: int,
) -> None:
    run_started_at = datetime.now().astimezone()
    solve_start = perf_counter()
    output_root = DEV_DATA_DIR / "pathway-pilot" / "output"
    staged_output_dir = staged_output_dir_for(output_dir, output_root)
    if staged_output_dir.exists():
        shutil.rmtree(staged_output_dir)
    staged_output_dir.mkdir(parents=True, exist_ok=True)

    data = build_model_inputs(
        capacity_factors=capacity_factors,
        demand=demand,
        periods=cfg.investment_periods,
        weather_year=weather_year,
        demand_zones=cfg.demand_zones,
        capacity_factor_zone=cfg.capacity_factor_zone,
        model_regions=cfg.model_regions,
    )
    network, status, condition = solve_model(cfg, data)
    run_time_seconds = perf_counter() - solve_start
    if (status, condition) != ("ok", "optimal"):
        raise RuntimeError(f"PyPSA solve failed: status={status}, condition={condition}")

    write_model_outputs(network, staged_output_dir)
    metadata = {
        "active_model": cfg.active_model,
        "demand_zones": list(cfg.demand_zones),
        "capacity_factor_zone": cfg.capacity_factor_zone,
        "model_regions": {
            name: {
                "demand_zones": list(region.demand_zones),
                "capacity_factor_zone": region.capacity_factor_zone,
            }
            for name, region in cfg.model_regions.items()
        },
        "interconnectors": [
            {
                "name": interconnector.name,
                "bus0": interconnector.bus0,
                "bus1": interconnector.bus1,
                "capacity_mw": interconnector.capacity_mw,
            }
            for interconnector in cfg.interconnectors
        ],
        "weather_year": weather_year,
        "run_at": run_started_at.isoformat(),
    }
    (staged_output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    with (staged_output_dir / "solve_times.txt").open("a", encoding="utf-8") as handle:
        handle.write(
            f"run_at: {run_started_at.isoformat()}, "
            f"run_time_secs: {run_time_seconds:.3f}\n"
        )
    copied_to_output = copy_output_bundle(staged_output_dir, output_dir)
    print(f"Wrote staged outputs to {staged_output_dir.resolve()}")
    if copied_to_output:
        print(f"Copied {cfg.active_model} weather {weather_year} model outputs to {output_dir}")
    print(f"Solve run time: {run_time_seconds:.3f} seconds")


def copy_output_bundle(staged_output_dir: Path, output_dir: Path) -> bool:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in staged_output_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, output_dir / path.name)
        return True
    except OSError as exc:
        print(f"Could not copy staged outputs to {output_dir}: {exc}")
        print(f"Staged outputs are preserved at {staged_output_dir.resolve()}")
        return False


def staged_output_dir_for(output_dir: Path, output_root: Path) -> Path:
    try:
        relative = output_dir.relative_to(output_root)
    except ValueError:
        relative = Path(output_dir.name)
    return STAGING_ROOT / relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one pathway-pilot PyPSA model case.")
    parser.add_argument("--model-case", help="Model case from config.model_cases, e.g. DK or NL.")
    parser.add_argument("--weather-year", type=int, help="Weather/climate year to solve.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path("config/model_config.yaml"))
    if args.model_case:
        cfg = with_active_model(cfg, args.model_case)
    weather_year = args.weather_year or cfg.climate_years[0]
    capacity_factors, demand = load_time_series()
    output_root = DEV_DATA_DIR / "pathway-pilot" / "output"
    output_dir = scenario_output_dir(
        output_root,
        cfg.active_model,
        weather_year,
        climate_year_count=len(cfg.climate_years),
    )
    run_model_case(cfg, capacity_factors, demand, output_dir, weather_year)


if __name__ == "__main__":
    main()
