from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_forced_wind_scenario import (  # noqa: E402
    STAGING_ROOT as FORCED_STAGING_ROOT,
    apply_exact_capacity_constraint,
    forced_scenario_name,
    generator_name,
    read_base_capacity_mw,
)
from run_pypsa_model import STAGING_ROOT as MODEL_STAGING_ROOT, staged_output_dir_for  # noqa: E402


def test_forced_scenario_names_target_generator():
    assert generator_name("DK", "wind", 2030) == "DK_wind_2030"
    assert (
        forced_scenario_name("DK_NL", "DK", "wind", 2030, 1000)
        == "DK_NL_forced_DK_wind_2030_plus_1000MW"
    )


def test_read_base_capacity_from_existing_output_shape():
    output_path = Path(".tmp") / "forced_wind_base_output"
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    pd.DataFrame(
        [
            {"generator": "DK_wind_2030", "p_nom_opt_mw": 1234.5},
            {"generator": "NL_wind_2030", "p_nom_opt_mw": 987.0},
        ]
    ).to_parquet(output_path / "optimal_capacities.parquet", index=False)

    assert read_base_capacity_mw(output_path, "DK_wind_2030") == 1234.5
    shutil.rmtree(output_path)


def test_apply_exact_capacity_constraint_sets_min_and_max():
    network = type("Network", (), {})()
    network.generators = pd.DataFrame(
        {"p_nom_min": [0.0], "p_nom_max": [5000.0]},
        index=["DK_wind_2030"],
    )

    apply_exact_capacity_constraint(network, "DK_wind_2030", 2234.5)

    assert network.generators.loc["DK_wind_2030", "p_nom_min"] == 2234.5
    assert network.generators.loc["DK_wind_2030", "p_nom_max"] == 2234.5


def test_forced_outputs_are_staged_inside_repo():
    assert not FORCED_STAGING_ROOT.is_absolute()
    assert FORCED_STAGING_ROOT.parts[0] == ".tmp"


def test_model_outputs_are_staged_inside_repo():
    output_root = Path(r"C:\Users\B510067\dev_data\pathway-pilot\output")
    output_dir = output_root / "DK_NL" / "weather_2008"

    staged = staged_output_dir_for(output_dir, output_root)

    assert not MODEL_STAGING_ROOT.is_absolute()
    assert MODEL_STAGING_ROOT.parts[0] == ".tmp"
    assert staged == MODEL_STAGING_ROOT / "DK_NL" / "weather_2008"
