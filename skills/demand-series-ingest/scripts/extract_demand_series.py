from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "target_year",
    "weather_year",
    "zone",
    "timestamp",
    "demand_mw",
    "source_file",
]


def read_demand_series(
    path: Path,
    target_year: int,
    zone: str,
    header: int = 7,
) -> pd.DataFrame:
    wide = pd.read_excel(path, sheet_name=zone, header=header)
    required = {"Date", "Month", "Day", "Hour"}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    weather_year_columns = [column for column in wide.columns if column not in required]
    if not weather_year_columns:
        raise ValueError("No weather-year columns found")

    long = wide.melt(
        id_vars=["Date", "Month", "Day", "Hour"],
        value_vars=weather_year_columns,
        var_name="weather_year",
        value_name="demand_mw",
    )
    long["weather_year"] = long["weather_year"].astype(float).astype("int16")
    long["target_year"] = int(target_year)
    long["zone"] = zone
    long["source_file"] = path.name
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
        description="Extract one hourly demand workbook sheet to long form."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target-year", required=True, type=int)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--header", type=int, default=7)
    args = parser.parse_args()

    table = read_demand_series(
        args.input,
        target_year=args.target_year,
        zone=args.zone,
        header=args.header,
    )
    write_table(table, args.output)
    print(table.head().to_string(index=False))
    print(f"Rows: {len(table):,}")
    print(f"Demand range: {table['demand_mw'].min():.2f} to {table['demand_mw'].max():.2f} MW")
    print(f"Annual demand by weather year:")
    print((table.groupby("weather_year")["demand_mw"].sum() / 1_000_000).head().to_string())
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
