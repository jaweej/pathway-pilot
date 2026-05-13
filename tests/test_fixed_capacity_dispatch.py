from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_fixed_capacity_dispatch import (  # noqa: E402
    STAGING_ROOT,
    apply_fixed_capacities,
    dispatch_scenario_name,
    fixed_capacity_map,
    single_period_config,
    staged_output_dir_for,
    target_generator_name,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathway_pilot.model_config import load_config  # noqa: E402


def test_dispatch_scenario_name_describes_source_and_target():
    assert (
        dispatch_scenario_name("DK", "DK_NL", 2008, 2030, "DK")
        == "DK_dispatch_fixed_2030_from_DK_NL_DK_weather_2008"
    )


def test_target_generator_name_strips_source_region_prefix():
    assert target_generator_name("DK_wind_2030", "DK") == "wind_2030"
    assert target_generator_name("wind_2030", None) == "wind_2030"


def test_single_period_config_keeps_only_requested_period():
    cfg = load_config(Path("config/model_config.yaml"))

    single = single_period_config(cfg, 2030)

    assert single.investment_periods == [2030]
    assert single.period_weights == {2030: cfg.period_weights[2030]}


def test_fixed_capacity_map_reads_source_region_capacities():
    output_path = Path(".tmp") / "fixed_dispatch_source_output"
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "generator": "DK_wind_2030",
                "bus": "DK",
                "carrier": "wind",
                "build_year": 2030,
                "p_nom_opt_mw": 123.0,
            },
            {
                "generator": "NL_wind_2030",
                "bus": "NL",
                "carrier": "wind",
                "build_year": 2030,
                "p_nom_opt_mw": 456.0,
            },
            {
                "generator": "DK_load_shedding",
                "bus": "DK",
                "carrier": "load_shedding",
                "build_year": 0,
                "p_nom_opt_mw": 999.0,
            },
        ]
    ).to_parquet(output_path / "optimal_capacities.parquet", index=False)

    capacities = fixed_capacity_map(output_path, "DK", 2030)

    assert capacities == {"wind_2030": 123.0}
    shutil.rmtree(output_path)


def test_apply_fixed_capacities_disables_investment_and_sets_nominal_capacity():
    network = type("Network", (), {})()
    network.generators = pd.DataFrame(
        {
            "carrier": ["wind", "gas", "load_shedding"],
            "p_nom_extendable": [True, True, False],
            "p_nom": [0.0, 0.0, 1000.0],
            "p_nom_min": [0.0, 0.0, 0.0],
            "p_nom_max": [5000.0, 5000.0, 0.0],
            "capital_cost": [1.0, 2.0, 0.0],
        },
        index=["wind_2030", "gas_turbine_2030", "load_shedding"],
    )

    apply_fixed_capacities(network, {"wind_2030": 123.0, "gas_turbine_2030": 45.0})

    assert network.generators.loc["wind_2030", "p_nom"] == 123.0
    assert network.generators.loc["gas_turbine_2030", "p_nom"] == 45.0
    assert not network.generators.loc["wind_2030", "p_nom_extendable"]
    assert network.generators.loc["gas_turbine_2030", "capital_cost"] == 0.0
    assert network.generators.loc["load_shedding", "p_nom"] == 1000.0


def test_dispatch_outputs_are_staged_inside_repo():
    output_root = Path(r"C:\Users\B510067\dev_data\pathway-pilot\output")
    output_dir = output_root / "DK_dispatch_fixed_2030_from_DK_NL_DK_weather_2008" / "weather_2008"

    staged = staged_output_dir_for(output_dir, output_root)

    assert not STAGING_ROOT.is_absolute()
    assert STAGING_ROOT.parts[0] == ".tmp"
    assert staged == STAGING_ROOT / "DK_dispatch_fixed_2030_from_DK_NL_DK_weather_2008" / "weather_2008"
