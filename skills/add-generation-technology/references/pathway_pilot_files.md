# Pathway-Pilot File Map

Core model files:

```text
config/model_config.yaml
src/pathway_pilot/model_inputs.py
src/pathway_pilot/technology_data.py
src/pathway_pilot/build_network.py
src/pathway_pilot/outputs.py
src/pathway_pilot/solve.py
```

Input-build scripts:

```text
scripts/build_techcat_gas_turbine_simple_cycle_large.py
scripts/build_techcat_wind_solar.py
scripts/build_pecd_capacity_factors.py
scripts/build_tyndp_demand_profiles.py
```

Run/report scripts:

```text
scripts/run_pypsa_model.py
scripts/build_output_dashboard.py
```

Tests to consider:

```text
tests/test_technology_data.py
tests/test_build_network.py
tests/test_model_inputs.py
tests/test_solve_smoke.py
tests/test_outputs.py
```

Docs:

```text
docs/pypsa_multiyear_pilot_spec.html
AGENTS.md
```

Related skills:

```text
skills/techcat-ingest
skills/capacity-factor-ingest
```
