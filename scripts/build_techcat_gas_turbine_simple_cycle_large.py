from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pathway_pilot.config import DEV_DATA_DIR

SHEET = "04 Gas turb. simple cycle, L"
XL_PATH = DEV_DATA_DIR / "techcat" / "technology_data_for_el_and_dh.xlsx"
OUT_PATH = DEV_DATA_DIR / "pathway-pilot" / "tech" / "gas_turbine_simple_cycle_large.parquet"

# Row indices (0-based) and ctrl-estimate year columns (2=2015, 3=2020, 4=2030, 5=2050)
YEARS = [2015, 2020, 2030, 2050]
CTRL_COLS = [2, 3, 4, 5]

ROWS = {
    "electrical_efficiency_net_annual_avg":   7,
    "technical_lifetime_years":              12,
    "total_nominal_investment_meur_per_mw_e": 27,
    "variable_om_eur_per_mwh_e":             30,
    "fixed_om_eur_per_mw_e_per_year":        31,
}


def main() -> None:
    raw = pd.read_excel(XL_PATH, sheet_name=SHEET, header=None)

    data: dict[str, list] = {"year": YEARS}
    for col_name, row_idx in ROWS.items():
        data[col_name] = [float(raw.iat[row_idx, c]) for c in CTRL_COLS]

    df = pd.DataFrame(data)

    # Interpolate 2040 as average of 2030 and 2050
    r2030 = df[df.year == 2030].iloc[0]
    r2050 = df[df.year == 2050].iloc[0]
    numeric_cols = [c for c in df.columns if c != "year"]
    row_2040 = {"year": 2040, **{c: (r2030[c] + r2050[c]) / 2 for c in numeric_cols}}
    df = pd.concat([df, pd.DataFrame([row_2040])], ignore_index=True)
    df = df.sort_values("year").reset_index(drop=True)

    df["year"] = df["year"].astype("int16")
    for c in numeric_cols:
        df[c] = df[c].astype("float32")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(df.to_string(index=False))
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
