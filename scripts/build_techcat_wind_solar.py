from __future__ import annotations

from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pathway_pilot.config import DEV_DATA_DIR

XL_PATH = DEV_DATA_DIR / "techcat" / "technology_data_for_el_and_dh.xlsx"
OUT_DIR = DEV_DATA_DIR / "pathway-pilot" / "tech"

SCHEMA_COLUMNS = [
    "year",
    "electrical_efficiency_net_annual_avg",
    "technical_lifetime_years",
    "total_nominal_investment_meur_per_mw_e",
    "variable_om_eur_per_mwh_e",
    "fixed_om_eur_per_mw_e_per_year",
]

TECHNOLOGIES = {
    "onshore_turbines": {
        "sheet": "20 Onshore turbines",
        "output": "onshore_turbines.parquet",
        "rows": {
            "technical_lifetime_years": 8,
            "total_nominal_investment_meur_per_mw_e": 15,
            "variable_om_eur_per_mwh_e": 22,
            "fixed_om_eur_per_mw_e_per_year": 23,
        },
    },
    "utility_scale_pv": {
        "sheet": "22 Utility-scale PV",
        "output": "utility_scale_pv.parquet",
        "rows": {
            "technical_lifetime_years": 8,
            "total_nominal_investment_meur_per_mw_e": 15,
            "fixed_om_eur_per_mw_e_per_year": 23,
        },
    },
}


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


def build_table(sheet: str, rows: dict[str, int]) -> pd.DataFrame:
    raw = pd.read_excel(XL_PATH, sheet_name=sheet, header=None)
    year_columns = _extract_year_columns(raw)

    data: dict[str, list[float | int]] = {"year": [year for year, _ in year_columns]}
    for column in SCHEMA_COLUMNS:
        if column == "year":
            continue
        row_idx = rows.get(column)
        if row_idx is None:
            data[column] = [math.nan for _ in year_columns]
            continue
        data[column] = [_float_or_nan(raw.iat[row_idx, col_idx]) for _, col_idx in year_columns]

    df = pd.DataFrame(data, columns=SCHEMA_COLUMNS)
    df["year"] = df["year"].astype("int16")
    for column in SCHEMA_COLUMNS:
        if column != "year":
            df[column] = df[column].astype("float32")
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for technology in TECHNOLOGIES.values():
        df = build_table(technology["sheet"], technology["rows"])
        output_path = OUT_DIR / technology["output"]
        df.to_parquet(output_path, index=False)
        print(f"\n{technology['sheet']}")
        print(df.to_string(index=False))
        print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
