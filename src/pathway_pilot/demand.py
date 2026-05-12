"""TYNDP demand profile processing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_YEARS = (2030, 2040)
ZONES = ("DKE1", "DKW1")


def reshape_demand_profile(
    wide: pd.DataFrame,
    target_year: int,
    zone: str,
    source_file: str,
) -> pd.DataFrame:
    weather_year_columns = [
        column for column in wide.columns if column not in ("Date", "Month", "Day", "Hour")
    ]
    long = wide.melt(
        id_vars=["Date", "Month", "Day", "Hour"],
        value_vars=weather_year_columns,
        var_name="weather_year",
        value_name="demand_mw",
    )
    long["weather_year"] = long["weather_year"].astype(float).astype("int16")
    long["target_year"] = target_year
    long["zone"] = zone
    long["source_file"] = source_file

    long["timestamp"] = pd.to_datetime(
        {
            "year": long["weather_year"].astype("int32"),
            "month": long["Month"].astype("int16") + 1,
            "day": long["Day"].astype("int16"),
            "hour": long["Hour"].astype("int16"),
        },
        errors="raise",
    )

    long["target_year"] = long["target_year"].astype("int16")
    long["demand_mw"] = long["demand_mw"].astype("float32")

    return long[
        [
            "target_year",
            "weather_year",
            "zone",
            "timestamp",
            "demand_mw",
            "source_file",
        ]
    ]


def read_demand_sheet(path: Path, target_year: int, zone: str) -> pd.DataFrame:
    wide = pd.read_excel(path, sheet_name=zone, header=7)
    return reshape_demand_profile(wide, target_year, zone, path.name)


def build_demand_table(input_dir: Path) -> pd.DataFrame:
    pieces = []
    for target_year in TARGET_YEARS:
        path = input_dir / f"{target_year}_National Trends.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"Missing required demand workbook: {path}")
        for zone in ZONES:
            pieces.append(read_demand_sheet(path, target_year, zone))
    return pd.concat(pieces, ignore_index=True)
