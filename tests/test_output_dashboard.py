from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_output_dashboard import _highest_shedding_week, load_dashboard_data


def test_highest_shedding_week_centers_peak_when_possible():
    rows = pd.DataFrame(
            {
                "timestep": pd.date_range("2030-01-01", periods=300, freq="h"),
                "load_shedding": [0.0] * 200 + [100.0] + [0.0] * 99,
            }
        )

    window = _highest_shedding_week(rows)

    peak_position = window.index[window["load_shedding"] > 0][0] - window.index[0]
    assert 80 <= peak_position <= 88
    assert len(window) == 168


def test_dashboard_loads_combined_model_regions_and_interconnector_flows():
    output_path = Path(".tmp") / "dashboard_test_output"
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    snapshots = pd.MultiIndex.from_product(
        [[2030], pd.date_range("2030-01-01", periods=2, freq="h")],
        names=["period", "timestep"],
    )
    capacities = pd.DataFrame(
        [
            {
                "generator": "DK_wind_2030",
                "bus": "DK",
                "carrier": "wind",
                "build_year": 2030,
                "lifetime": 30,
                "p_nom_opt_mw": 100,
                "capital_cost": 1,
                "unit_capex_eur_per_mw": 1,
                "marginal_cost": 0,
            },
            {
                "generator": "DK_load_shedding",
                "bus": "DK",
                "carrier": "load_shedding",
                "build_year": 0,
                "lifetime": 999,
                "p_nom_opt_mw": 1000,
                "capital_cost": 0,
                "unit_capex_eur_per_mw": 0,
                "marginal_cost": 10000,
            },
            {
                "generator": "NL_wind_2030",
                "bus": "NL",
                "carrier": "wind",
                "build_year": 2030,
                "lifetime": 30,
                "p_nom_opt_mw": 100,
                "capital_cost": 1,
                "unit_capex_eur_per_mw": 1,
                "marginal_cost": 0,
            },
            {
                "generator": "NL_load_shedding",
                "bus": "NL",
                "carrier": "load_shedding",
                "build_year": 0,
                "lifetime": 999,
                "p_nom_opt_mw": 1000,
                "capital_cost": 0,
                "unit_capex_eur_per_mw": 0,
                "marginal_cost": 10000,
            },
        ]
    )
    dispatch = pd.DataFrame(
        [
            {
                "period": period,
                "timestep": timestep,
                "generator": generator,
                "dispatch_mw": value,
                "bus": bus,
                "carrier": carrier,
                "build_year": 2030,
            }
            for period, timestep in snapshots
            for generator, bus, carrier, value in [
                ("DK_wind_2030", "DK", "wind", 70),
                ("DK_load_shedding", "DK", "load_shedding", 0),
                ("NL_wind_2030", "NL", "wind", 60),
                ("NL_load_shedding", "NL", "load_shedding", 0),
            ]
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "period": period,
                "timestep": timestep,
                "bus": bus,
                "price_eur_per_mwh": 50 if bus == "DK" else 60,
            }
            for period, timestep in snapshots
            for bus in ["DK", "NL"]
        ]
    )
    flows = pd.DataFrame(
        [
            {
                "period": period,
                "timestep": timestep,
                "link": "DK_NL_interconnector",
                "bus0": "DK",
                "bus1": "NL",
                "flow_bus0_to_bus1_mw": 10 if i == 0 else -5,
            }
            for i, (period, timestep) in enumerate(snapshots)
        ]
    )
    metadata = {
        "active_model": "DK_NL",
        "weather_year": 2008,
        "model_regions": {
            "DK": {"demand_zones": ["DKE1", "DKW1"], "capacity_factor_zone": "DKW1"},
            "NL": {"demand_zones": ["NL00"], "capacity_factor_zone": "NL00"},
        },
        "interconnectors": [
            {
                "name": "DK_NL_interconnector",
                "bus0": "DK",
                "bus1": "NL",
                "capacity_mw": 1000,
            }
        ],
    }

    capacities.to_parquet(output_path / "optimal_capacities.parquet", index=False)
    dispatch.to_parquet(output_path / "hourly_dispatch.parquet", index=False)
    prices.to_parquet(output_path / "hourly_prices.parquet", index=False)
    flows.to_parquet(output_path / "hourly_interconnector_flows.parquet", index=False)
    (output_path / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    compact_data = load_dashboard_data(output_path)
    data = load_dashboard_data(output_path, compact_week_data=False)

    assert sorted(data["regions"]) == ["DK", "NL"]
    assert compact_data["regions"]["DK"]["weekDataMode"] == "compact"
    assert set(compact_data["regions"]["DK"]["weekData"]["2030"]) == {"winter", "summer", "shed"}
    dk_week = data["regions"]["DK"]["weekData"]["2030"]
    nl_week = data["regions"]["NL"]["weekData"]["2030"]
    assert data["regions"]["DK"]["dispatchCarriers"][-3:] == [
        "interconnector_import",
        "interconnector_export",
        "load_shedding",
    ]
    assert dk_week[0]["interconnector_export"] == -10
    assert dk_week[1]["interconnector_import"] == 5
    assert nl_week[0]["interconnector_import"] == 10
    assert nl_week[1]["interconnector_export"] == -5
    assert data["regions"]["DK"]["security"][0]["import_twh"] == 0.0
    assert data["regions"]["DK"]["security"][0]["export_twh"] == 0.0
    assert data["regions"]["NL"]["security"][0]["import_twh"] == 0.0
    assert data["regions"]["NL"]["security"][0]["export_twh"] == 0.0
    assert data["regions"]["DK"]["security"][0]["interconnector_utilisation_pct"] == 0.75
    shutil.rmtree(output_path)
