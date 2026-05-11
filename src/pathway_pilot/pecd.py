"""PECD capacity-factor processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TARGET_YEARS = (2030, 2040)
ZONES = ("DKE1", "DKW1")

TECHNOLOGY_FILE_PREFIXES = {
    "onshore_wind": "PECD_Wind_Onshore",
    "solar": "PECD_LFSolarPV",
}


@dataclass(frozen=True)
class PecdInput:
    target_year: int
    zone: str
    technology: str
    path: Path


def expected_inputs(pecd_dir: Path) -> list[PecdInput]:
    inputs: list[PecdInput] = []
    for target_year in TARGET_YEARS:
        for zone in ZONES:
            for technology, prefix in TECHNOLOGY_FILE_PREFIXES.items():
                path = (
                    pecd_dir
                    / str(target_year)
                    / f"{prefix}_{target_year}_{zone}_edition 2023.2.csv"
                )
                inputs.append(PecdInput(target_year, zone, technology, path))
    return inputs


def validate_inputs(inputs: list[PecdInput]) -> None:
    missing = [item.path for item in inputs if not item.path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required PECD input files:\n{missing_text}")


def read_pecd_capacity_factors(input_file: PecdInput) -> pd.DataFrame:
    wide = pd.read_csv(input_file.path, skiprows=10)
    weather_year_columns = [column for column in wide.columns if column not in ("Date", "Hour")]

    long = wide.melt(
        id_vars=["Date", "Hour"],
        value_vars=weather_year_columns,
        var_name="weather_year",
        value_name="capacity_factor",
    )
    long["weather_year"] = long["weather_year"].astype(float).astype("int16")
    long["target_year"] = input_file.target_year
    long["zone"] = input_file.zone
    long["technology"] = input_file.technology
    long["source_file"] = input_file.path.name

    date_parts = long["Date"].str.extract(r"(?P<day>\d{2})\.(?P<month>\d{2})\.")
    long["timestamp"] = pd.to_datetime(
        {
            "year": long["weather_year"].astype("int32"),
            "month": date_parts["month"].astype("int16"),
            "day": date_parts["day"].astype("int16"),
            "hour": long["Hour"].astype("int16") - 1,
        },
        errors="raise",
    )

    long["target_year"] = long["target_year"].astype("int16")
    long["capacity_factor"] = long["capacity_factor"].astype("float32")

    return long[
        [
            "target_year",
            "weather_year",
            "zone",
            "technology",
            "timestamp",
            "capacity_factor",
            "source_file",
        ]
    ]


def build_capacity_factor_table(pecd_dir: Path) -> pd.DataFrame:
    inputs = expected_inputs(pecd_dir)
    validate_inputs(inputs)
    return pd.concat(
        [read_pecd_capacity_factors(input_file) for input_file in inputs],
        ignore_index=True,
    )
