# AGENTS.md

## How to install
- Use the local virtualenv: `python -m venv .venv`
- Install with pip: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
- Do not use `uv` for this repo.

## How to run tests
- Run all tests: `.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider`
- PyPSA solve tests must use tiny model instances with very few hours.
- Run the large model only when explicitly needed: `.\.venv\Scripts\python.exe scripts\run_pypsa_model.py`

## Key architecture
- `src/pathway_pilot/model_inputs.py` prepares synthetic and real demand/CF time series.
- `src/pathway_pilot/technology_data.py` adapts TechCat costs and gas prices.
- `src/pathway_pilot/build_network.py` builds the PyPSA network.
- `src/pathway_pilot/solve.py` runs HiGHS via PyPSA.
- `src/pathway_pilot/outputs.py` writes capacities, dispatch, and prices.
- `scripts/` contains data-build, model-run, and dashboard entry points.

## Coding conventions
- Prefer small, typed, testable functions in `src/pathway_pilot`.
- Keep model-facing tables long-form unless a library API needs wide data.
- Keep PyPSA optimization costs in EUR/MW/year or EUR/MWh as appropriate.
- Keep reporting CAPEX as overnight unit CAPEX in EUR/MW or MEUR/MW.
- Use `apply_patch` for manual edits and avoid unrelated refactors.

## Never touch these files
- Do not edit `.venv/`.
- Do not edit raw files under `C:\Users\B510067\dev_data\TYNDP24`.
- Do not edit raw TechCat Excel files under `C:\Users\B510067\dev_data\techcat`.
- Do not overwrite generated files in `C:\Users\B510067\dev_data\pathway-pilot` unless the task explicitly asks to regenerate them.
- Do not revert user changes or unrelated dirty worktree files.

## Data/security constraints
- Keep raw data and generated model outputs outside git under `DEV_DATA_DIR`.
- Default `DEV_DATA_DIR` is `C:\Users\B510067\dev_data`; override with `PATHWAY_PILOT_DEV_DATA_DIR`.
- Do not commit proprietary/raw input data, solver outputs, dashboards, or local logs.
- Treat gas prices configured in DKK/GJ as fuel prices and convert using `EURDKK = 7.47`.
- Select DK/NL with `active_model` in `config/model_config.yaml`; DK uses DKW1 CFs, NL uses NL00 CFs; reuse 2040 time series for 2050 only for demand and capacity factors.

## Definition of done
- Relevant tests pass.
- Large model runs are optimal when model outputs are changed; use `scripts\run_pypsa_scenarios.py` for DK/NL scenario output folders.
- Dashboard is regenerated when output schema or reporting logic changes.
- Spec/docs are updated when behavior or assumptions change.
- New outputs are read back or spot-checked for units and expected values.
