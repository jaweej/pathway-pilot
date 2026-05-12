import pandas as pd

from pathway_pilot.demand import ZONES, reshape_demand_profile


def test_demand_source_zones_include_nl00():
    assert ZONES == ("DKE1", "DKW1", "NL00")


def test_reshape_demand_profile_matches_long_schema():
    wide = pd.DataFrame(
        {
            "Date": [1, 1],
            "Month": [0, 0],
            "Day": [1, 1],
            "Hour": [0, 1],
            "1982": [10.0, 11.0],
            "1983": [20.0, 21.0],
        }
    )

    long = reshape_demand_profile(
        wide,
        target_year=2030,
        zone="DKE1",
        source_file="2030_National Trends.xlsx",
    )

    assert list(long.columns) == [
        "target_year",
        "weather_year",
        "zone",
        "timestamp",
        "demand_mw",
        "source_file",
    ]
    assert len(long) == 4
    assert long.loc[0, "timestamp"] == pd.Timestamp("1982-01-01 00:00")
    first_1983 = long[long["weather_year"] == 1983].iloc[0]
    assert first_1983["timestamp"] == pd.Timestamp("1983-01-01 00:00")
