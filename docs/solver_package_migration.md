# Solver package migration

The generic GAMS-bundled CPLEX bridge and its PyPSA-Eur integration were moved
to the separate internal repository:

```text
https://gitlab.ens.dk/ffk/linopy-gams-cplex.git
```

`pathway-pilot` remains the DK/NL multiperiod investment application. It
depends on the standalone package through `requirements.txt` and retains only
the PyPSA-specific model construction, solution assignment, metadata, and
output workflows.

The standalone repository owns:

- GAMS discovery and licence probing;
- the direct CPLEX callable-library route;
- the legacy MPS-to-GAMS fallback;
- generic Linopy tests and diagnostics;
- the pinned PyPSA-Eur tutorial and 50-cluster/12-hour integration examples.

No GAMS binaries, licence files, PyPSA-Eur datasets, proprietary input data, or
solver results are committed to either repository.
