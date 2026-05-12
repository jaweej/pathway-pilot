import pandas as pd

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
        for zone in ["DKE1", "DKW1"]:
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
                        "demand_mw": 10 if zone == "DKE1" else 20 + i,
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
