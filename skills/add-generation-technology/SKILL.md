---
name: add-generation-technology
description: Add a new generation technology to the pathway-pilot PyPSA capacity-expansion model. Use when Codex needs to wire a new generator technology through TechCat/input data, config capacity limits, technology assumptions, PyPSA network construction, outputs/dashboard reporting, tests, and spec updates.
---

# Add Generation Technology

## Overview

Use this skill when adding a generator technology beyond the current wind, solar, gas turbine, and load-shedding setup. Keep the change narrow: data ingestion, assumptions, network component creation, tests, and reporting.

## First Classify The Technology

- **Variable renewable:** needs hourly capacity factors by model period and zone.
- **Thermal fuelled:** needs efficiency, fuel price treatment, variable O&M, fixed O&M, and CAPEX.
- **Dispatchable non-fuel:** needs CAPEX, fixed O&M, variable O&M, and any availability rule.
- **Load shedding / slack:** treat separately; do not mix with normal generation.

## Implementation Checklist

1. Add or confirm the processed technology data file under `DEV_DATA_DIR\pathway-pilot\tech`.
2. Add the technology key and file name in `src/pathway_pilot/technology_data.py`.
3. Add fallback assumptions only if missing source files should still allow tests to run.
4. Add the capacity limit in `config/model_config.yaml`.
5. Add a PyPSA carrier and one extendable generator per investment period in `src/pathway_pilot/build_network.py`.
6. Attach `unit_capex_eur_per_mw` for reporting.
7. If variable renewable, add capacity-factor series to `ModelInputs` or map from existing real-data tables.
8. Update output/dashboard grouping if the carrier should appear in charts.
9. Add focused tests using tiny hour counts.
10. Update `docs/pypsa_multiyear_pilot_spec.html`.

## Cost Rules

- PyPSA `capital_cost` must be annualized EUR/MW/year.
- Reporting `unit_capex_eur_per_mw` must be overnight CAPEX from TechCat.
- Thermal marginal cost should be fuel cost divided by efficiency plus variable O&M.
- Gas prices are configured in DKK/GJ fuel and converted to EUR/MWh fuel with `EURDKK = 7.47`.
- Do not reuse 2040 technology costs for 2050 if a 2050 TechCat row exists.

## Time-Series Rules

- Existing model-facing demand is DKE1 + DKW1.
- Existing wind and solar capacity factors use DKW1.
- Reuse 2040 time series for 2050 only for demand and capacity factors.
- Keep raw prepared capacity-factor files as source years; apply model-period reuse in `model_inputs.py`.

## Test Expectations

- Unit-test technology assumptions and marginal-cost formulas.
- Unit-test network components and carrier/generator names.
- Smoke-test solving only with tiny model instances and very few hours.
- Test output columns if adding reporting fields.

## Definition Of Done

- `.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider` passes.
- Large model is rerun only if requested or output behavior changes.
- Dashboard is regenerated if carriers, outputs, or reporting charts change.
- New generated data is spot-checked but not committed.

## References

Read `references/pathway_pilot_files.md` for the file-by-file edit map.
