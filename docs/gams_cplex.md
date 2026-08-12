# Solving with GAMS-bundled CPLEX

The pathway model can solve its continuous PyPSA/Linopy linear program with the CPLEX callable library shipped in a locally licensed GAMS installation.

The default, fast transfer path:

1. builds the normal PyPSA/Linopy model;
2. writes the sparse linear program to MPS;
3. loads that MPS directly into the CPLEX callable library;
4. solves with CPLEX barrier using six threads by default, following POMATO's barrier and multithreaded configuration pattern; and
5. transfers primal values, constraint marginals, and the objective directly back to Linopy so PyPSA can populate its normal result tables.

This avoids constructing a second, generic GAMS algebraic model. The previous bridge remains available as a fallback; it converts MPS to GDX/GAMS with `mps2gms`, expands the generic matrix with the GAMS model generator, invokes the GAMS/CPLEX link, and reads the solution back from GDX.

The fast implementation is in `src/pathway_pilot/cplex_callable.py`; the legacy bridge is in `src/pathway_pilot/gams_cplex.py`. Both currently support continuous LPs, which is the formulation used by this model.

## Run one case

From the repository root, the fast transfer mode is the default for `gams-cplex`:

```powershell
.\.venv\Scripts\python.exe scripts\run_pypsa_model.py `
  --model-case DK_NL `
  --weather-year 2009 `
  --solver gams-cplex
```

Use the old GAMS model-generation path only when it is specifically needed:

```powershell
--cplex-transfer gams
```

The default solver settings can be overridden for controlled comparisons:

```powershell
--cplex-method dual --cplex-threads 1
```

Available methods are `automatic`, `primal`, `dual`, `network`, `barrier`, `sifting`, and `concurrent`.

The adapter checks `PATHWAY_PILOT_GAMS_DIR`, then `PATH`, then installations under `C:\GAMS`. Before selecting an installation it runs a tiny model through the licensed GAMS/CPLEX link. On the current workstation this selects `C:\GAMS\37` with CPLEX 20.1; the newer GAMS 49 installation is beyond the installed license's maintenance date. An installation can also be selected explicitly:

```powershell
--gams-dir C:\GAMS\37
```

Use `--output-root` to keep comparison results separate from the normal output tree:

```powershell
--output-root .tmp\solver_validation\cplex
```

No additional Python packages are required. Temporary MPS and legacy GDX/GAMS files are staged below `.tmp\gams_cplex` and removed after solution read-back.

## Performance

Full 2009-weather-year runs were timed on the same workstation and inputs on 2026-08-12.

| Model | Legacy GAMS route, total | Direct MPS export + CPLEX read | Direct CPLEX solve | Direct route, total | Overall speedup |
|---|---:|---:|---:|---:|---:|
| DK | 62.915 s | 3.688 s | 14.100 s | 24.884 s | **2.53x** |
| DK_NL | 7,415.876 s | 7.795 s | 31.798 s | 47.545 s | **155.98x** |

The legacy route did not expose a reliable separate generator timer, so an exact construction-to-construction ratio is not claimed. During that run, however, the generic GAMS model generator was the long-lived bottleneck. Its replacement, the directly measured MPS export and CPLEX load, takes 7.795 seconds for DK_NL. The construction bottleneck is therefore removed, and the requested roughly tenfold end-to-end improvement is exceeded by a wide margin.

For context, a fresh HiGHS run of the same DK_NL case took 556.857 seconds. The optimized direct CPLEX route is about 11.71 times faster than HiGHS on this case. POMATO's CPLEX settings were material: an initial conservative validation using single-thread dual simplex took 726.804 seconds end to end, while barrier with six threads reduced that to 47.545 seconds.

Each run records `solver`, `cplex_transfer`, `cplex_method`, `cplex_threads`, `solver_timings_seconds`, and `objective` in `model_metadata.json`.

## DK_NL 2009 numerical validation

The optimized direct CPLEX result was compared both with the legacy GAMS/CPLEX result and with a fresh HiGHS solve from identical inputs.

| Measure | Direct versus legacy CPLEX | Direct CPLEX versus HiGHS |
|---|---:|---:|
| Reconstructed objective | absolute difference below `2e-4 EUR` | identical to displayed precision |
| Optimal capacity by generator | max `5.09e-11 MW` | max `8.64e-11 MW` |
| Hourly dispatch aggregated system-wide by carrier | max `1.02e-10 MW` | max `1.02e-10 MW` |
| Hourly marginal price | max `5.82e-9 EUR/MWh` | max `4.08e-8 EUR/MWh` |
| Total load shedding | difference below `2e-11 MWh` | difference below `6e-11 MWh` |

Individual interconnector flows can differ by up to 2,000 MW between optimal solutions. The link is lossless and costless, so alternative flow directions and regional dispatch allocations can be mathematically interchangeable. System-wide carrier dispatch, objective, capacities, prices, and energy balance agree to numerical tolerance; the flow difference is LP degeneracy rather than a different economic result.

## Licensing note

The fast path deliberately selects only a GAMS installation that first passes a real licensed GAMS/CPLEX solve, and then loads the CPLEX library distributed with that installation. This technical check does not determine contractual entitlement to call the library outside the GAMS solver link. GAMS states that use of the GAMS/CPLEX link is subject to appropriate licensing and a valid IBM license agreement. Confirm that the organization's GAMS/IBM terms cover callable-library use before relying on the direct mode in production. See the [official GAMS CPLEX documentation](https://gams.com/latest/docs/S_CPLEX.html) and [GAMS licensing documentation](https://gams.com/latest/docs/UG_License.html).
