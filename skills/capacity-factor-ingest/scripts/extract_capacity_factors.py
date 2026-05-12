from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "target_year",
    "weather_year",
    "zone",
    "technology",
    "timestamp",
    "capacity_factor",
    "source_file",
]


def read_capacity_factors(
    path: Path,
    target_year: int,
    zone: str,
    technology: str,
    skiprows: int = 10,
) -> pd.DataFrame:
    wide = pd.read_csv(path, skiprows=skiprows)
    required = {"Date", "Hour"}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    weather_year_columns = [column for column in wide.columns if column not in required]
    if not weather_year_columns:
        raise ValueError("No weather-year columns found")

    long = wide.melt(
        id_vars=["Date", "Hour"],
        value_vars=weather_year_columns,
        var_name="weather_year",
        value_name="capacity_factor",
    )
    long["weather_year"] = long["weather_year"].astype(float).astype("int16")
    long["target_year"] = int(target_year)
    long["zone"] = zone
    long["technology"] = technology
    long["source_file"] = path.name

    date_parts = long["Date"].astype(str).str.extract(r"(?P<day>\d{2})\.(?P<month>\d{2})\.")
    if date_parts.isna().any().any():
        raise ValueError("Could not parse Date values as DD.MM.")

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
    return long[OUTPUT_COLUMNS]


def write_table(table: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".parquet":
        table.to_parquet(output, index=False)
    elif suffix == ".csv":
        table.to_csv(output, index=False)
    else:
        raise ValueError("Output must end with .parquet or .csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one PECD-style hourly capacity-factor CSV to long form."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target-year", required=True, type=int)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--technology", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skiprows", type=int, default=10)
    args = parser.parse_args()

    table = read_capacity_factors(
        args.input,
        target_year=args.target_year,
        zone=args.zone,
        technology=args.technology,
        skiprows=args.skiprows,
    )
    write_table(table, args.output)
    print(table.head().to_string(index=False))
    print(f"Rows: {len(table):,}")
    print(
        "Capacity factor range: "
        f"{table['capacity_factor'].min():.4f} to {table['capacity_factor'].max():.4f}"
    )
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
