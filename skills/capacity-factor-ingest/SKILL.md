---
name: capacity-factor-ingest
description: Ingest hourly technology capacity factors by technology, target year, weather year, and price area into model-ready long-form tables. Use when Codex needs to process PECD-style CSV files or similar hourly renewable availability files for wind, solar, or other technologies and zones in pathway-pilot.
---

# Capacity Factor Ingest

## Overview

Use this skill to turn hourly capacity-factor source files into the pathway-pilot long-form schema. Prefer the repo's existing `pathway_pilot.pecd` module for PECD data, and use the bundled helper for one-off checks or new PECD-style sources.

## Workflow

1. Locate source files under `DEV_DATA_DIR\TYNDP24\PECD\PECD`.
2. Confirm technology, target year, price area, weather-year columns, and timestamp fields.
3. Convert wide weather-year columns to long form.
4. Build timestamps from weather year, date, and hour.
5. Keep capacity factors unitless, normally in `[0, 1]`.
6. Save processed files outside git under `DEV_DATA_DIR\pathway-pilot\tvar`.
7. Spot-check row counts, min/max capacity factor, zones, technologies, and years.

## Standard Schema

Use this output schema:

```text
target_year
weather_year
zone
technology
timestamp
capacity_factor
source_file
```

`timestamp` should represent the weather-year timestamp, not the target-year planning period.

## Pathway-Pilot Rules

- Current model input uses the configured `model_cases[active_model].capacity_factor_zone`.
- Current `DK` uses `DKW1`; current `NL` uses `NL00`.
- Current prepared PECD data covers target years 2030 and 2040.
- For model period 2050, reuse the 2040 capacity-factor time series only at the model-input adapter layer.
- Do not bake the 2050 reuse rule into raw capacity-factor ingestion files.
- Keep technology names stable with repo code: `onshore_wind` and `solar`.

## Existing Repo Entry Points

Use these before writing new ingestion code:

```text
src/pathway_pilot/pecd.py
scripts/build_pecd_capacity_factors.py
```

The builder writes:

```text
DEV_DATA_DIR\pathway-pilot\tvar\pecd_capacity_factors_2030_2040.parquet
```

## Bundled Helper

Use `scripts/extract_capacity_factors.py` for one PECD-style file:

```powershell
python skills\capacity-factor-ingest\scripts\extract_capacity_factors.py `
  --input C:\Users\B510067\dev_data\TYNDP24\PECD\PECD\2030\PECD_LFSolarPV_2030_DKW1_edition 2023.2.csv `
  --target-year 2030 `
  --zone DKW1 `
  --technology solar `
  --output .tmp\solar_dkw1_2030.parquet
```

The helper assumes PECD headers with `Date`, `Hour`, and one column per weather year after `--skiprows 10`.

## Validation

- Check `capacity_factor` min/max; values outside `[0, 1]` need explanation.
- Check each source file produces `8760 * number_of_weather_years` rows unless leap days or source gaps are documented.
- Check output includes expected zones and technologies before using it in model inputs.
- Keep raw data and processed outputs out of git.
