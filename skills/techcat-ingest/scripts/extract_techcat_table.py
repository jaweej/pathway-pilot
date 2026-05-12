from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


SCHEMA_COLUMNS = [
    "year",
    "electrical_efficiency_net_annual_avg",
    "technical_lifetime_years",
    "total_nominal_investment_meur_per_mw_e",
    "variable_om_eur_per_mwh_e",
    "fixed_om_eur_per_mw_e_per_year",
]


def _parse_row_mapping(values: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --row name=index, got {value!r}")
        name, index = value.split("=", 1)
        if name not in SCHEMA_COLUMNS:
            raise ValueError(f"Unknown schema column {name!r}")
        if name == "year":
            raise ValueError("Do not pass a row mapping for year")
        mapping[name] = int(index)
    return mapping


def _extract_year_columns(raw: pd.DataFrame) -> list[tuple[int, int]]:
    year_row = raw.iloc[1]
    estimate_row = raw.iloc[2]
    year_columns: list[tuple[int, int]] = []
    for col_idx, value in year_row.items():
        if str(estimate_row[col_idx]).strip().lower() != "ctrl":
            continue
        if pd.isna(value):
            continue
        try:
            year = int(value)
        except (TypeError, ValueError):
            continue
        year_columns.append((year, int(col_idx)))
    return year_columns


def _float_or_nan(value: object) -> float:
    if pd.isna(value):
        return math.nan
    return float(value)


def build_table(xlsx: Path, sheet: str, rows: dict[str, int]) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, sheet_name=sheet, header=None)
    year_columns = _extract_year_columns(raw)
    if not year_columns:
        raise ValueError("No ctrl year columns found in workbook header")

    data: dict[str, list[float | int]] = {"year": [year for year, _ in year_columns]}
    for column in SCHEMA_COLUMNS:
        if column == "year":
            continue
        row_idx = rows.get(column)
        if row_idx is None:
            data[column] = [math.nan for _ in year_columns]
        else:
            data[column] = [_float_or_nan(raw.iat[row_idx, col_idx]) for _, col_idx in year_columns]

    table = pd.DataFrame(data, columns=SCHEMA_COLUMNS)
    table["year"] = table["year"].astype("int16")
    for column in SCHEMA_COLUMNS:
        if column != "year":
            table[column] = table[column].astype("float32")
    return table


def add_interpolated_year(
    table: pd.DataFrame,
    year: int,
    from_year: int,
    to_year: int,
) -> pd.DataFrame:
    if year in set(table["year"]):
        return table.sort_values("year").reset_index(drop=True)

    start = table.loc[table["year"] == from_year]
    end = table.loc[table["year"] == to_year]
    if start.empty or end.empty:
        raise ValueError(f"Cannot interpolate {year}: missing {from_year} or {to_year}")

    start_row = start.iloc[0]
    end_row = end.iloc[0]
    weight = (year - from_year) / (to_year - from_year)
    new_row = {"year": year}
    for column in SCHEMA_COLUMNS:
        if column == "year":
            continue
        new_row[column] = start_row[column] + (end_row[column] - start_row[column]) * weight

    return (
        pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)
        .sort_values("year")
        .reset_index(drop=True)
        .astype(table.dtypes.to_dict())
    )


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
    parser = argparse.ArgumentParser(description="Extract a TechCat sheet to a schema-stable table.")
    parser.add_argument("--xlsx", required=True, type=Path)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--row", action="append", default=[], help="Metric row mapping as column_name=zero_based_row_index")
    parser.add_argument("--interpolate-year", type=int)
    parser.add_argument("--from-year", type=int)
    parser.add_argument("--to-year", type=int)
    args = parser.parse_args()

    table = build_table(args.xlsx, args.sheet, _parse_row_mapping(args.row))
    if args.interpolate_year is not None:
        if args.from_year is None or args.to_year is None:
            raise ValueError("--interpolate-year requires --from-year and --to-year")
        table = add_interpolated_year(table, args.interpolate_year, args.from_year, args.to_year)
    write_table(table, args.output)
    print(table.to_string(index=False))
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
