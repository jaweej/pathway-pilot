---
name: demand-series-ingest
description: Ingest hourly power demand data series by target year, weather year, and price area into model-ready long-form tables. Use when Codex needs to process TYNDP demand profile Excel workbooks or similar hourly demand files for DKE1, DKW1, or other price areas in pathway-pilot.
---

# Demand Series Ingest

## Overview

Use this skill to convert hourly electricity demand workbooks into the pathway-pilot long-form schema. Prefer the repo's existing `pathway_pilot.demand` module for TYNDP National Trends data, and use the bundled helper for one-off checks or new price-area sheets.

## Workflow

1. Locate source workbooks under `DEV_DATA_DIR\TYNDP24\Demand\Demand Profiles\NT\Electricity demand profiles`.
2. Confirm target year, price-area sheet name, weather-year columns, and timestamp columns.
3. Convert wide weather-year columns to long form.
4. Build timestamps from weather year, month, day, and hour.
5. Keep demand in MW.
6. Save processed files outside git under `DEV_DATA_DIR\pathway-pilot\tvar`.
7. Spot-check row counts, zones, years, peak demand, and annual demand.

## Standard Schema

Use this output schema:

```text
target_year
weather_year
zone
timestamp
demand_mw
source_file
```

`timestamp` should represent the weather-year timestamp, not the target-year planning period.

## Pathway-Pilot Rules

- Current source zones are `DKE1` and `DKW1`.
- Current source target years are `2030` and `2040`.
- Current model-facing demand is the hourly sum of `DKE1` and `DKW1`.
- For model period `2050`, reuse the `2040` demand time series only at the model-input adapter layer.
- Do not bake the 2050 reuse rule into raw demand ingestion files.

## Existing Repo Entry Points

Use these before writing new ingestion code:

```text
src/pathway_pilot/demand.py
scripts/build_tyndp_demand_profiles.py
```

The builder writes:

```text
DEV_DATA_DIR\pathway-pilot\tvar\tyndp_demand_profiles_2030_2040.parquet
```

## Bundled Helper

Use `scripts/extract_demand_series.py` for one workbook sheet:

```powershell
.\.venv\Scripts\python.exe skills\demand-series-ingest\scripts\extract_demand_series.py `
  --input "C:\Users\B510067\dev_data\TYNDP24\Demand\Demand Profiles\NT\Electricity demand profiles\2030_National Trends.xlsx" `
  --target-year 2030 `
  --zone DKE1 `
  --output .tmp\dke1_2030_demand.parquet
```

The helper assumes TYNDP workbooks where the data table starts at Excel header row 8, matching `pandas.read_excel(..., header=7)`.

## Validation

- Check each zone/year produces `8760 * number_of_weather_years` rows unless source gaps are documented.
- Check `demand_mw` is non-negative.
- Check peak demand and annual demand are plausible for the price area.
- Check weather-year timestamps parse without missing values.
- Keep raw data and processed outputs out of git.
