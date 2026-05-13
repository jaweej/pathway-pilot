from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pathway_pilot.build_network import build_network
from pathway_pilot.config import DEV_DATA_DIR
from pathway_pilot.model_config import ModelConfig, load_config, with_active_model
from pathway_pilot.model_inputs import build_model_inputs
from pathway_pilot.outputs import write_model_outputs

from run_pypsa_model import copy_output_bundle, load_time_series, scenario_output_dir


STAGING_ROOT = Path(".tmp") / "dispatch_model_outputs"


def dispatch_scenario_name(
    target_model_case: str,
    source_model_case: str,
    source_weather_year: int,
    period: int,
    source_region: str | None,
) -> str:
    source_label = source_model_case if source_region is None else f"{source_model_case}_{source_region}"
    return (
        f"{target_model_case}_dispatch_fixed_{period}_"
        f"from_{source_label}_weather_{source_weather_year}"
    )


def single_period_config(cfg: ModelConfig, period: int) -> ModelConfig:
    if period not in cfg.investment_periods:
        raise ValueError(f"Period {period} is not in configured investment periods")
    return replace(
        cfg,
        investment_periods=[period],
        period_weights={period: cfg.period_weights[period]},
    )


def target_generator_name(source_generator: str, source_region: str | None) -> str:
    if source_region is None:
        return source_generator
    prefix = f"{source_region}_"
    if not source_generator.startswith(prefix):
        raise ValueError(f"Source generator {source_generator!r} is not in region {source_region!r}")
    return source_generator.removeprefix(prefix)


def fixed_capacity_map(
    source_output_dir: Path,
    source_region: str | None,
    period: int,
) -> dict[str, float]:
    capacities_path = source_output_dir / "optimal_capacities.parquet"
    if not capacities_path.exists():
        raise FileNotFoundError(f"Source capacities not found: {capacities_path}")
    capacities = pd.read_parquet(capacities_path)
    capacities = capacities[
        (capacities["build_year"] == period)
        & (~capacities["carrier"].isin(["load_shedding"]))
    ]
    if source_region is not None:
        if "bus" in capacities.columns:
            capacities = capacities[capacities["bus"] == source_region]
        else:
            capacities = capacities[
                capacities["generator"].astype(str).str.startswith(f"{source_region}_")
            ]
    result = {
        target_generator_name(str(row.generator), source_region): float(row.p_nom_opt_mw)
        for row in capacities.itertuples(index=False)
    }
    if not result:
        raise ValueError(f"No fixed capacities found in {capacities_path}")
    return result


def apply_fixed_capacities(network, capacities_mw: dict[str, float]) -> None:
    missing = sorted(set(capacities_mw) - set(network.generators.index))
    if missing:
        raise ValueError(f"Fixed capacity generators not found in target network: {missing}")

    dispatchable = network.generators["carrier"] != "load_shedding"
    network.generators.loc[dispatchable, "p_nom_extendable"] = False
    network.generators.loc[dispatchable, "p_nom"] = 0.0
    network.generators.loc[dispatchable, "p_nom_min"] = 0.0
    network.generators.loc[dispatchable, "p_nom_max"] = 0.0
    network.generators.loc[dispatchable, "capital_cost"] = 0.0
    for generator, capacity_mw in capacities_mw.items():
        network.generators.loc[generator, "p_nom"] = capacity_mw


def solve_dispatch_network(network, solver_name: str = "highs") -> tuple[str, str]:
    return network.optimize(
        multi_investment_periods=True,
        solver_name=solver_name,
        include_objective_constant=False,
    )


def staged_output_dir_for(output_dir: Path, output_root: Path) -> Path:
    try:
        relative = output_dir.relative_to(output_root)
    except ValueError:
        relative = Path(output_dir.name)
    return STAGING_ROOT / relative


def run_fixed_capacity_dispatch(
    cfg: ModelConfig,
    target_model_case: str,
    source_model_case: str,
    source_weather_year: int,
    target_weather_year: int,
    period: int,
    source_region: str | None,
    output_root: Path,
) -> Path:
    source_output_dir = scenario_output_dir(
        output_root,
        source_model_case,
        source_weather_year,
        climate_year_count=len(cfg.climate_years),
    )
    capacities_mw = fixed_capacity_map(source_output_dir, source_region, period)
    scenario_name = dispatch_scenario_name(
        target_model_case,
        source_model_case,
        source_weather_year,
        period,
        source_region,
    )
    output_dir = scenario_output_dir(
        output_root,
        scenario_name,
        target_weather_year,
        climate_year_count=len(cfg.climate_years),
    )
    staged_output_dir = staged_output_dir_for(output_dir, output_root)
    if staged_output_dir.exists():
        shutil.rmtree(staged_output_dir)
    staged_output_dir.mkdir(parents=True, exist_ok=True)

    target_cfg = single_period_config(with_active_model(cfg, target_model_case), period)
    capacity_factors, demand = load_time_series()
    data = build_model_inputs(
        capacity_factors=capacity_factors,
        demand=demand,
        periods=target_cfg.investment_periods,
        weather_year=target_weather_year,
        demand_zones=target_cfg.demand_zones,
        capacity_factor_zone=target_cfg.capacity_factor_zone,
        model_regions=target_cfg.model_regions,
    )
    network = build_network(target_cfg, data)
    apply_fixed_capacities(network, capacities_mw)

    run_started_at = datetime.now().astimezone()
    solve_start = perf_counter()
    status, condition = solve_dispatch_network(network)
    run_time_seconds = perf_counter() - solve_start
    if (status, condition) != ("ok", "optimal"):
        raise RuntimeError(f"PyPSA dispatch solve failed: status={status}, condition={condition}")

    write_model_outputs(network, staged_output_dir)
    metadata = {
        "active_model": scenario_name,
        "base_model_case": source_model_case,
        "dispatch_model": True,
        "target_model_case": target_model_case,
        "source_model_case": source_model_case,
        "source_region": source_region,
        "source_weather_year": source_weather_year,
        "source_output_dir": str(source_output_dir),
        "fixed_capacity_period": period,
        "weather_year": target_weather_year,
        "run_at": run_started_at.isoformat(),
        "demand_zones": list(target_cfg.demand_zones),
        "capacity_factor_zone": target_cfg.capacity_factor_zone,
        "model_regions": {
            name: {
                "demand_zones": list(region.demand_zones),
                "capacity_factor_zone": region.capacity_factor_zone,
            }
            for name, region in target_cfg.model_regions.items()
        },
        "interconnectors": [],
        "fixed_capacities_mw": capacities_mw,
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
    print(f"Read fixed capacities from {source_output_dir}")
    for generator, capacity_mw in sorted(capacities_mw.items()):
        print(f"Fixed {generator}: {capacity_mw:.3f} MW")
    print(f"Wrote staged outputs to {staged_output_dir.resolve()}")
    if copied_to_output:
        print(f"Copied {scenario_name} weather {target_weather_year} outputs to {output_dir}")
    print(f"Solve run time: {run_time_seconds:.3f} seconds")
    return output_dir if copied_to_output else staged_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dispatch-only model with capacities fixed from an existing output."
    )
    parser.add_argument("--target-model-case", default="DK")
    parser.add_argument("--source-model-case", default="DK_NL")
    parser.add_argument("--source-region", default="DK")
    parser.add_argument("--source-weather-year", type=int, default=2008)
    parser.add_argument("--target-weather-year", type=int, default=2008)
    parser.add_argument("--period", type=int, default=2030)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path("config/model_config.yaml"))
    output_root = DEV_DATA_DIR / "pathway-pilot" / "output"
    run_fixed_capacity_dispatch(
        cfg=cfg,
        target_model_case=args.target_model_case,
        source_model_case=args.source_model_case,
        source_weather_year=args.source_weather_year,
        target_weather_year=args.target_weather_year,
        period=args.period,
        source_region=args.source_region or None,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
