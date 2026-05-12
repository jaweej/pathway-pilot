---
name: techcat-ingest
description: Extract technology data from TechCat Excel workbooks into clean tabular files. Use when Codex needs to ingest efficiencies, technical lifetimes, CAPEX, variable OPEX, fixed OPEX, or related cost/performance rows from a TechCat .xlsx file, especially for pathway-pilot technology Parquet inputs.
---

# TechCat Ingest

## Overview

Use this skill to convert TechCat XLSX sheets into model-ready tables with consistent names, units, and year rows. Prefer deterministic parsing with the bundled script when the sheet layout matches the Danish Energy Agency TechCat workbook style.

## Workflow

1. Locate the workbook, usually `DEV_DATA_DIR\techcat\technology_data_for_el_and_dh.xlsx`.
2. Identify the technology sheet names and row indices for the needed metrics.
3. Confirm year columns from the workbook header: year row plus `ctrl` estimate row.
4. Extract to long-ish technology tables with one row per year.
5. Preserve source units in column names.
6. Validate values against the workbook for at least one year and one technology.
7. Save processed repo data outside git, usually under `DEV_DATA_DIR\pathway-pilot\tech`.

## Standard Schema

Use these columns when building pathway-pilot technology files:

```text
year
electrical_efficiency_net_annual_avg
technical_lifetime_years
total_nominal_investment_meur_per_mw_e
variable_om_eur_per_mwh_e
fixed_om_eur_per_mw_e_per_year
```

Use `NaN` for metrics that do not apply, such as PV electrical efficiency or missing variable O&M.

## Units

- Keep TechCat overnight CAPEX as `total_nominal_investment_meur_per_mw_e`.
- Convert to EUR/MW only in model adapters: multiply MEUR/MW by `1_000_000`.
- Keep fixed OPEX as EUR/MW/year.
- Keep variable OPEX as EUR/MWh_e.
- Keep efficiencies as fractions if the workbook stores fractions; check before converting percentages.

## Bundled Script

Use `scripts/extract_techcat_table.py` for repeatable extraction:

```powershell
python C:\Users\B510067\.codex\skills\techcat-ingest\scripts\extract_techcat_table.py `
  --xlsx C:\Users\B510067\dev_data\techcat\technology_data_for_el_and_dh.xlsx `
  --sheet "22 Utility-scale PV" `
  --output C:\Users\B510067\dev_data\pathway-pilot\tech\utility_scale_pv.parquet `
  --row technical_lifetime_years=8 `
  --row total_nominal_investment_meur_per_mw_e=15 `
  --row fixed_om_eur_per_mw_e_per_year=23
```

Add repeated `--row name=index` arguments for each metric. Row indices are zero-based, matching pandas `header=None`.

For TechCat sheets with 2030 and 2050 control columns but no 2040 column, add:

```powershell
--interpolate-year 2040 --from-year 2030 --to-year 2050
```

## Pathway-Pilot Notes

- Use the repo's existing scripts as examples before inventing a new layout:
  - `scripts/build_techcat_gas_turbine_simple_cycle_large.py`
  - `scripts/build_techcat_wind_solar.py`
- Technology costs should use the actual TechCat row for each model year when available.
- Do not reuse 2040 costs for 2050; only demand and capacity-factor time series have that reuse rule.
- PyPSA `capital_cost` is annualized EUR/MW/year. Dashboard/reporting unit CAPEX should come from overnight TechCat CAPEX.

## Validation

- Print or inspect extracted rows for 2030, 2040, and 2050.
- Check solar 2030 utility-scale PV CAPEX is about `0.38 MEUR/MW` if using the current TechCat workbook.
- Confirm generated files are not committed if they live under `DEV_DATA_DIR`.
