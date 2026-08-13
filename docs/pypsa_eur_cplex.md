# PyPSA-Eur with the direct CPLEX solver

The official PyPSA-Eur `v2026.02.0` checkout is installed in `pypsa-eur/`.
The checked-in integration pins upstream commit
`d6383ebf602767b1adbb676fe8a16e37a6e9f932` and supports two instances:

- `tutorial`: Belgium, five spatial clusters, 1-7 March 2013, aggregated to
  seven daily snapshots.
- `elec-50-12h`: the default full-Europe electricity model, 50 electrical
  clusters, the full 2013 weather year, and 12-hour aggregation (730
  snapshots).

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
.\.venv\Scripts\python.exe scripts\run_pypsa_eur_cplex.py --instance elec-50-12h
```

The setup script clones the pinned release when needed, creates the external
storage layout, applies the versioned CPLEX patch, renders the selected local
config, retrieves its data, builds the network, and solves it. The default
instance remains `tutorial`. Useful partial modes are `--setup-only` and
`--prepare-only`; they can be combined with either instance.

Override the default `C:\Users\B510067\dev_data` root with
`PATHWAY_PILOT_DEV_DATA_DIR`. The current config uses the licensed GAMS 37
installation at `C:\GAMS\37`, CPLEX barrier, and six threads.

The output network is:

```text
<DEV_DATA_DIR>/pathway-pilot/pypsa-eur-v2026.02.0/results/
  cplex-tutorial/networks/base_s_5_elec_.nc
```

For the 50-node instance it is:

```text
<DEV_DATA_DIR>/pathway-pilot/pypsa-eur-v2026.02.0/results/
  cplex-elec-50-12h/networks/base_s_50_elec_12h.nc
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

## Verified 50-node run (2026-08-12)

The full-Europe `elec-50-12h` result reloaded successfully with 730 snapshots,
397 generators, 98 AC lines, and 261 links. The requested 50-cluster workflow
produced 49 electricity buses after PyPSA-Eur's clustering cleanup. PyPSA
reports 147 buses in total because it also creates 49 hydrogen and 49 battery
buses.
CPLEX returned `optimal` with an objective of `51,656,806,942.62714 EUR/year`.

| Stage | Time |
|---|---:|
| Linopy MPS export | 13.208 s |
| CPLEX MPS read | 1.641 s |
| CPLEX optimization | 1,048.547 s |
| Complete solve rule | 1,091.46 s |

The saved dispatch contains no missing generator, line, or link values. An
independent nodal-balance audit found a maximum residual of `1.2e-4 MW` over
all buses and snapshots. Peak solve-rule resident memory was about 3.21 GB.
The one-time input build uses a 6.14 GB official Europe weather cutout; reruns
reuse all downloaded data, availability matrices, and renewable profiles.
