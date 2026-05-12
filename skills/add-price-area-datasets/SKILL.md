---
name: add-price-area-datasets
description: Coordinate demand-series and capacity-factor ingestion to add one or more additional price areas to pathway-pilot datasets. Use when Codex needs to extend model-ready demand and renewable availability inputs for a new price area, update model case configuration to reference that area, or validate that both demand and capacity-factor coverage exist for a price-area scenario.
---

# Add Price Area Datasets

## Overview

Use this skill to add a price area consistently across the pathway-pilot input stack. It composes `$demand-series-ingest` for hourly demand and `$capacity-factor-ingest` for hourly renewable capacity factors, then checks that the resulting datasets can be selected through `config/model_config.yaml`.

## Workflow

1. Identify the price area code, target years, weather years, and whether the task needs demand, capacity factors, or both.
2. Read and follow `skills/demand-series-ingest/SKILL.md` for demand workbooks and `skills/capacity-factor-ingest/SKILL.md` for PECD or PECD-style capacity-factor files.
3. Locate raw inputs under `DEV_DATA_DIR`, normally:
   - Demand: `TYNDP24\Demand\Demand Profiles\NT\Electricity demand profiles`
   - Capacity factors: `TYNDP24\PECD\PECD`
4. Prefer existing repo entry points before writing new ingestion code:
   - Demand: `src/pathway_pilot/demand.py`, `scripts/build_tyndp_demand_profiles.py`
   - Capacity factors: `src/pathway_pilot/pecd.py`, `scripts/build_pecd_capacity_factors.py`
5. Extend parser mappings or builder configuration only as needed for the new price area. Keep changes small and typed.
6. Write processed outputs outside git under `DEV_DATA_DIR\pathway-pilot\tvar`; do not commit raw data or generated parquet files.
7. If the new price area should be model-selectable, update `config/model_config.yaml` `model_cases` with:
   - `demand_zones`: one or more demand zones to aggregate
   - `capacity_factor_zone`: the PECD capacity-factor zone
8. Update tests for the new zone mappings, aggregation behavior, and configuration validation.

## Dataset Rules

- Keep demand long-form with `target_year`, `weather_year`, `zone`, `timestamp`, `demand_mw`, and `source_file`.
- Keep capacity factors long-form with `target_year`, `weather_year`, `zone`, `technology`, `timestamp`, `capacity_factor`, and `source_file`.
- Keep source target years as source target years. If period `2050` reuses `2040`, apply that only in model input adapters, not in raw ingestion files.
- Keep capacity factors unitless and normally in `[0, 1]`; keep demand in MW.
- Preserve existing technology names expected by the model: `onshore_wind` and `solar`.

## Validation

- Confirm every requested price area appears in both processed tables when both demand and capacity factors are required.
- Check row counts for each zone, target year, weather year, and technology. Expect `8760` rows per weather year unless leap days or source gaps are documented.
- Spot-check demand annual energy, peak demand, non-negative values, and timestamp parsing.
- Spot-check capacity-factor min/max, technology coverage, and timestamp parsing.
- Run focused tests first, then the repo test command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

## When To Stop

Stop and ask for source-file clarification when the price area exists in demand but not PECD capacity-factor files, when the source uses a new file layout, or when a price-area code maps ambiguously to multiple demand or capacity-factor zones.
