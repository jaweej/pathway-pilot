"""Time-series input adapters for the pathway pilot model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelInputs:
    demand_series: pd.Series
    wind_cf_series: pd.Series
    solar_cf_series: pd.Series

    @property
    def snapshots(self) -> pd.MultiIndex:
        return self.demand_series.index


def _snapshot_index(periods: list[int], hours_per_period: int) -> pd.MultiIndex:
    tuples = []
    for period in periods:
        timestamps = pd.date_range(f"{period}-01-01", periods=hours_per_period, freq="h")
        tuples.extend((period, timestamp) for timestamp in timestamps)
    return pd.MultiIndex.from_tuples(tuples, names=["period", "timestep"])


def make_synthetic_inputs(periods: list[int], hours_per_period: int = 8760) -> ModelInputs:
    snapshots = _snapshot_index(periods, hours_per_period)
    hour = np.arange(len(snapshots))
    day_fraction = (hour % 24) / 24
    year_fraction = (hour % hours_per_period) / max(hours_per_period, 1)

    demand = 900 + 180 * np.sin(2 * np.pi * day_fraction) + 70 * np.cos(
        2 * np.pi * year_fraction
    )
    wind = 0.45 + 0.18 * np.sin(2 * np.pi * (year_fraction + 0.15))
    solar = np.maximum(0, np.sin(np.pi * day_fraction)) * (
        0.55 + 0.15 * np.sin(2 * np.pi * year_fraction)
    )

    return ModelInputs(
        demand_series=pd.Series(demand, index=snapshots, name="demand_mw").astype("float32"),
        wind_cf_series=pd.Series(np.clip(wind, 0, 1), index=snapshots, name="wind").astype(
            "float32"
        ),
        solar_cf_series=pd.Series(np.clip(solar, 0, 1), index=snapshots, name="solar").astype(
            "float32"
        ),
    )


def _with_model_period(frame: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    pieces = []
    for period in periods:
        source_year = 2040 if period == 2050 else period
        piece = frame[frame["target_year"] == source_year].copy()
        piece["period"] = period
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _series_from_frame(frame: pd.DataFrame, value_column: str) -> pd.Series:
    index = pd.MultiIndex.from_frame(frame[["period", "timestamp"]])
    index.names = ["period", "timestep"]
    return pd.Series(frame[value_column].to_numpy(), index=index, name=value_column).sort_index()


def build_model_inputs(
    capacity_factors: pd.DataFrame,
    demand: pd.DataFrame,
    periods: list[int],
    weather_year: int,
) -> ModelInputs:
    cf = capacity_factors[capacity_factors["weather_year"] == weather_year].copy()
    cf = cf[cf["zone"] == "DKW1"]
    cf = _with_model_period(cf, periods)

    demand_year = demand[demand["weather_year"] == weather_year].copy()
    demand_year = _with_model_period(demand_year, periods)
    demand_agg = (
        demand_year.groupby(["period", "timestamp"], as_index=False)["demand_mw"].sum()
    )

    wind = _series_from_frame(cf[cf["technology"] == "onshore_wind"], "capacity_factor")
    solar = _series_from_frame(cf[cf["technology"] == "solar"], "capacity_factor")
    demand_series = _series_from_frame(demand_agg, "demand_mw")

    return ModelInputs(
        demand_series=demand_series.astype("float32"),
        wind_cf_series=wind.reindex(demand_series.index).astype("float32"),
        solar_cf_series=solar.reindex(demand_series.index).astype("float32"),
    )
