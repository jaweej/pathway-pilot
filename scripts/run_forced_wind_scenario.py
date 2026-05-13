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

from pathway_pilot.build_network import build_network
from pathway_pilot.config import DEV_DATA_DIR
from pathway_pilot.model_config import ModelConfig, load_config, with_active_model
from pathway_pilot.model_inputs import build_model_inputs
from pathway_pilot.outputs import write_model_outputs

from run_pypsa_model import load_time_series, scenario_output_dir


STAGING_ROOT = Path(".tmp") / "forced_scenario_outputs"


def forced_scenario_name(
    model_case: str,
    region: str,
    technology: str,
    build_year: int,
    extra_mw: float,
) -> str:
    extra_label = f"{extra_mw:g}".replace(".", "p")
    return f"{model_case}_forced_{region}_{technology}_{build_year}_plus_{extra_label}MW"


def generator_name(region: str, technology: str, build_year: int) -> str:
    return f"{region}_{technology}_{build_year}"


def read_base_capacity_mw(base_output_dir: Path, generator: str) -> float:
    capacities_path = base_output_dir / "optimal_capacities.parquet"
    if not capacities_path.exists():
        raise FileNotFoundError(f"Base capacities not found: {capacities_path}")
    capacities = pd.read_parquet(capacities_path)
    rows = capacities[capacities["generator"] == generator]
    if rows.empty:
        raise ValueError(f"Generator {generator!r} not found in {capacities_path}")
    return float(rows.iloc[0]["p_nom_opt_mw"])


def apply_exact_capacity_constraint(
    network,
    generator: str,
    forced_capacity_mw: float,
) -> None:
    if generator not in network.generators.index:
        raise ValueError(f"Generator {generator!r} not found in network")
    p_nom_max = float(network.generators.loc[generator, "p_nom_max"])
    if forced_capacity_mw > p_nom_max:
        raise ValueError(
            f"Forced capacity {forced_capacity_mw:.3f} MW exceeds "
            f"{generator} p_nom_max {p_nom_max:.3f} MW"
        )
    network.generators.loc[generator, "p_nom_min"] = forced_capacity_mw
    network.generators.loc[generator, "p_nom_max"] = forced_capacity_mw


def solve_network(network, solver_name: str = "highs") -> tuple[str, str]:
    return network.optimize(
        multi_investment_periods=True,
        solver_name=solver_name,
        include_objective_constant=False,
    )


def metadata_for_forced_run(
    cfg: ModelConfig,
    scenario_name: str,
    base_model_case: str,
    weather_year: int,
    generator: str,
    base_capacity_mw: float,
    extra_mw: float,
    forced_capacity_mw: float,
    base_output_dir: Path,
    run_started_at: datetime,
) -> dict:
    return {
        "active_model": scenario_name,
        "base_model_case": base_model_case,
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
        "forced_capacity": {
            "generator": generator,
            "base_capacity_mw": base_capacity_mw,
            "extra_mw": extra_mw,
            "forced_capacity_mw": forced_capacity_mw,
            "constraint": "p_nom_min = p_nom_max = forced_capacity_mw",
            "base_output_dir": str(base_output_dir),
        },
    }


def run_forced_wind_scenario(
    cfg: ModelConfig,
    model_case: str,
    weather_year: int,
    region: str,
    technology: str,
    build_year: int,
    extra_mw: float,
    output_root: Path,
) -> Path:
    target_generator = generator_name(region, technology, build_year)
    base_output_dir = scenario_output_dir(
        output_root,
        model_case,
        weather_year,
        climate_year_count=len(cfg.climate_years),
    )
    base_capacity_mw = read_base_capacity_mw(base_output_dir, target_generator)
    forced_capacity_mw = base_capacity_mw + extra_mw
    scenario_name = forced_scenario_name(model_case, region, technology, build_year, extra_mw)
    output_dir = scenario_output_dir(
        output_root,
        scenario_name,
        weather_year,
        climate_year_count=len(cfg.climate_years),
    )
    staged_output_dir = scenario_output_dir(
        STAGING_ROOT,
        scenario_name,
        weather_year,
        climate_year_count=len(cfg.climate_years),
    )
    if staged_output_dir.exists():
        shutil.rmtree(staged_output_dir)
    staged_output_dir.mkdir(parents=True, exist_ok=True)

    capacity_factors, demand = load_time_series()
    data = build_model_inputs(
        capacity_factors=capacity_factors,
        demand=demand,
        periods=cfg.investment_periods,
        weather_year=weather_year,
        demand_zones=cfg.demand_zones,
        capacity_factor_zone=cfg.capacity_factor_zone,
        model_regions=cfg.model_regions,
    )
    network = build_network(cfg, data)
    apply_exact_capacity_constraint(network, target_generator, forced_capacity_mw)

    run_started_at = datetime.now().astimezone()
    solve_start = perf_counter()
    status, condition = solve_network(network)
    run_time_seconds = perf_counter() - solve_start
    if (status, condition) != ("ok", "optimal"):
        raise RuntimeError(f"PyPSA solve failed: status={status}, condition={condition}")

    write_model_outputs(network, staged_output_dir)
    metadata = metadata_for_forced_run(
        cfg=cfg,
        scenario_name=scenario_name,
        base_model_case=model_case,
        weather_year=weather_year,
        generator=target_generator,
        base_capacity_mw=base_capacity_mw,
        extra_mw=extra_mw,
        forced_capacity_mw=forced_capacity_mw,
        base_output_dir=base_output_dir,
        run_started_at=run_started_at,
    )
    (staged_output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    with (staged_output_dir / "solve_times.txt").open("a", encoding="utf-8") as handle:
        handle.write(
            f"run_at: {run_started_at.isoformat()}, "
            f"run_time_secs: {run_time_seconds:.3f}\n"
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in staged_output_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, output_dir / path.name)
        copied_to_output = True
    except OSError as exc:
        copied_to_output = False
        print(f"Could not copy staged outputs to {output_dir}: {exc}")
        print(f"Staged outputs are preserved at {staged_output_dir.resolve()}")

    print(f"Read base {target_generator}: {base_capacity_mw:.3f} MW from {base_output_dir}")
    print(f"Forced {target_generator}: {forced_capacity_mw:.3f} MW")
    print(f"Wrote staged outputs to {staged_output_dir.resolve()}")
    if copied_to_output:
        print(f"Copied {scenario_name} weather {weather_year} model outputs to {output_dir}")
    print(f"Solve run time: {run_time_seconds:.3f} seconds")
    return output_dir if copied_to_output else staged_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a forced wind scenario using an existing unrestricted model output "
            "as the capacity reference."
        )
    )
    parser.add_argument("--model-case", default="DK_NL")
    parser.add_argument("--weather-year", type=int, default=2008)
    parser.add_argument("--region", default="DK")
    parser.add_argument("--technology", default="wind")
    parser.add_argument("--build-year", type=int, default=2030)
    parser.add_argument("--extra-mw", type=float, default=1000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = with_active_model(load_config(Path("config/model_config.yaml")), args.model_case)
    output_root = DEV_DATA_DIR / "pathway-pilot" / "output"
    run_forced_wind_scenario(
        cfg=cfg,
        model_case=args.model_case,
        weather_year=args.weather_year,
        region=args.region,
        technology=args.technology,
        build_year=args.build_year,
        extra_mw=args.extra_mw,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
