import pandas as pd

from pathway_pilot.model_config import ModelRegion
from pathway_pilot.model_inputs import build_model_inputs, make_synthetic_inputs


def test_synthetic_hourly_data_shape():
    data = make_synthetic_inputs(periods=[2030, 2040, 2050], hours_per_period=8760)

    assert len(data.demand_series) == 3 * 8760
    assert data.wind_cf_series.between(0, 1).all()
    assert data.solar_cf_series.between(0, 1).all()


def test_real_data_adapter_uses_dkw1_cfs_summed_demand_and_2040_for_2050():
    timestamps_2030 = pd.date_range("1982-01-01", periods=2, freq="h")
    timestamps_2040 = pd.date_range("1982-01-01", periods=2, freq="h")
    cf_rows = []
    demand_rows = []
    for target_year, timestamps in [(2030, timestamps_2030), (2040, timestamps_2040)]:
        for zone in ["DKE1", "DKW1", "NL00"]:
            for technology in ["onshore_wind", "solar"]:
                for i, timestamp in enumerate(timestamps):
                    cf_rows.append(
                        {
                            "target_year": target_year,
                            "weather_year": 1982,
                            "zone": zone,
                            "technology": technology,
                            "timestamp": timestamp,
                            "capacity_factor": 0.1 if zone == "DKE1" else 0.2 + i / 10,
                            "source_file": "cf.csv",
                        }
                    )
            for i, timestamp in enumerate(timestamps):
                demand_rows.append(
                    {
                        "target_year": target_year,
                        "weather_year": 1982,
                        "zone": zone,
                        "timestamp": timestamp,
                        "demand_mw": {"DKE1": 10, "DKW1": 20 + i, "NL00": 1000}[zone],
                        "source_file": "demand.xlsx",
                    }
                )

    data = build_model_inputs(
        capacity_factors=pd.DataFrame(cf_rows),
        demand=pd.DataFrame(demand_rows),
        periods=[2030, 2040, 2050],
        weather_year=1982,
    )

    assert data.wind_cf_series.loc[(2030, timestamps_2030[0])] == 0.2
    assert data.solar_cf_series.loc[(2030, timestamps_2030[1])] == 0.3
    assert data.demand_series.loc[(2030, timestamps_2030[1])] == 31
    assert data.wind_cf_series.loc[(2050, timestamps_2040[0])] == 0.2
    assert data.demand_series.loc[(2050, timestamps_2040[1])] == 31


def test_real_data_adapter_can_switch_to_nl_demand():
    timestamps = pd.date_range("1982-01-01", periods=2, freq="h")
    cf_rows = []
    demand_rows = []
    for zone in ["DKE1", "DKW1", "NL00"]:
        for technology in ["onshore_wind", "solar"]:
            for i, timestamp in enumerate(timestamps):
                cf_rows.append(
                    {
                        "target_year": 2030,
                        "weather_year": 1982,
                        "zone": zone,
                        "technology": technology,
                        "timestamp": timestamp,
                        "capacity_factor": {"DKE1": 0.1, "DKW1": 0.2, "NL00": 0.4}[zone]
                        + i / 10,
                        "source_file": "cf.csv",
                    }
                )
    for zone, value in [("DKE1", 10), ("DKW1", 20), ("NL00", 100)]:
        for timestamp in timestamps:
            demand_rows.append(
                {
                    "target_year": 2030,
                    "weather_year": 1982,
                    "zone": zone,
                    "timestamp": timestamp,
                    "demand_mw": value,
                    "source_file": "demand.xlsx",
                }
            )

    data = build_model_inputs(
        capacity_factors=pd.DataFrame(cf_rows),
        demand=pd.DataFrame(demand_rows),
        periods=[2030],
        weather_year=1982,
        demand_zones=("NL00",),
        capacity_factor_zone="NL00",
    )

    assert data.demand_series.loc[(2030, timestamps[0])] == 100
    assert data.wind_cf_series.loc[(2030, timestamps[1])] == 0.5


def test_real_data_adapter_can_build_dk_nl_bus_tables():
    timestamps = pd.date_range("1982-01-01", periods=2, freq="h")
    cf_rows = []
    demand_rows = []
    for zone in ["DKE1", "DKW1", "NL00"]:
        for technology in ["onshore_wind", "solar"]:
            for i, timestamp in enumerate(timestamps):
                cf_rows.append(
                    {
                        "target_year": 2030,
                        "weather_year": 1982,
                        "zone": zone,
                        "technology": technology,
                        "timestamp": timestamp,
                        "capacity_factor": {"DKE1": 0.1, "DKW1": 0.2, "NL00": 0.4}[zone]
                        + i / 10,
                        "source_file": "cf.csv",
                    }
                )
        for timestamp in timestamps:
            demand_rows.append(
                {
                    "target_year": 2030,
                    "weather_year": 1982,
                    "zone": zone,
                    "timestamp": timestamp,
                    "demand_mw": {"DKE1": 10, "DKW1": 20, "NL00": 100}[zone],
                    "source_file": "demand.xlsx",
                }
            )

    data = build_model_inputs(
        capacity_factors=pd.DataFrame(cf_rows),
        demand=pd.DataFrame(demand_rows),
        periods=[2030],
        weather_year=1982,
        model_regions={
            "DK": ModelRegion(demand_zones=("DKE1", "DKW1"), capacity_factor_zone="DKW1"),
            "NL": ModelRegion(demand_zones=("NL00",), capacity_factor_zone="NL00"),
        },
    )

    assert data.bus_names == ("DK", "NL")
    assert data.demand_by_bus.loc[(2030, timestamps[0]), "DK"] == 30
    assert data.demand_by_bus.loc[(2030, timestamps[0]), "NL"] == 100
    assert data.wind_cf_by_bus.loc[(2030, timestamps[1]), "DK"] == 0.3
    assert data.wind_cf_by_bus.loc[(2030, timestamps[1]), "NL"] == 0.5
    assert data.demand_series.loc[(2030, timestamps[0])] == 130
