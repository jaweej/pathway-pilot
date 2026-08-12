# PyPSA-Eur with the direct CPLEX solver

The official PyPSA-Eur `v2026.02.0` checkout is installed in `pypsa-eur/`.
The checked-in integration pins upstream commit
`d6383ebf602767b1adbb676fe8a16e37a6e9f932` and runs the official electricity
tutorial: Belgium, five spatial clusters, 1–7 March 2013, aggregated to seven
daily snapshots. This is the supported small tutorial workflow, not a
continent-wide production scenario.

All downloaded inputs and generated workflow artifacts are stored below:

```text
<DEV_DATA_DIR>/pathway-pilot/pypsa-eur-v2026.02.0/
```

The checkout's dataset namespaces, `cutouts`, `resources`, `results`, `logs`,
and `benchmarks` directories are Windows junctions into this data root. Static
configuration files distributed in the upstream repository remain in the
checkout.

## Install and run

Use the repository-local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-pypsa-eur.txt
.\.venv\Scripts\python.exe scripts\run_pypsa_eur_cplex.py
```

The setup script clones the pinned release when needed, creates the external
storage layout, applies the versioned CPLEX patch, renders the local config,
retrieves the tutorial data, builds the network, and solves it. Useful partial
modes are `--setup-only` and `--prepare-only`.

Override the default `C:\Users\B510067\dev_data` root with
`PATHWAY_PILOT_DEV_DATA_DIR`. The current config uses the licensed GAMS 37
installation at `C:\GAMS\37`, CPLEX barrier, and six threads.

The output network is:

```text
<DEV_DATA_DIR>/pathway-pilot/pypsa-eur-v2026.02.0/results/
  cplex-tutorial/networks/base_s_5_elec_.nc
```

Its `pathway_pilot_cplex` metadata records MPS export, CPLEX read, model
transfer, and solver times. CPLEX is called through the fast callable-library
bridge; HiGHS is used only by Linopy's MPS writer and does not optimize the
network.

## Verified run (2026-08-12)

The saved network reloaded successfully with 10 buses (five electricity buses
plus carrier buses), seven snapshots, 30 generators, six lines, and 546 LP
variables. CPLEX returned `optimal` with an objective of
`54,925,289.39364673 EUR/year`.

| Stage | Time |
|---|---:|
| Linopy MPS export | 0.314 s |
| CPLEX MPS read | 0.026 s |
| CPLEX optimization | 0.036 s |
| Complete solve rule | 33 s |

A separately constructed, identically seeded model solved with tight HiGHS
simplex settings produced an objective only `2.24e-8 EUR/year` different from
CPLEX. Maximum differences were `5.64e-11 MW` for generator capacity and
`4.55e-12 MVA` for line capacity. A raw-matrix audit also found maximum primal
constraint residuals of about `2e-11` for both solvers.

The full first build downloaded about 2.1 GB and took roughly 23 minutes,
including one resume after installing the initially missing Dask distributed
component. Subsequent runs reuse the external data and prepared network.
