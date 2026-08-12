"""Solve Linopy linear programs with CPLEX through a licensed GAMS installation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import xarray as xr


class GamsCplexError(RuntimeError):
    """Raised when the GAMS/CPLEX adapter cannot produce a valid solution."""


@dataclass(frozen=True)
class GamsInstallation:
    """Executables required by the MPS-to-GAMS solver bridge."""

    directory: Path
    gams: Path
    mps2gms: Path
    gdxdump: Path


def _installation_from_path(path: str | Path) -> GamsInstallation | None:
    candidate = Path(path)
    directory = candidate.parent if candidate.name.lower() == "gams.exe" else candidate
    installation = GamsInstallation(
        directory=directory,
        gams=directory / "gams.exe",
        mps2gms=directory / "mps2gms.exe",
        gdxdump=directory / "gdxdump.exe",
    )
    if all(
        executable.is_file()
        for executable in (installation.gams, installation.mps2gms, installation.gdxdump)
    ):
        return installation
    return None


def _version_key(path: Path) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", path.parent.name)
    return tuple(int(number) for number in numbers) or (0,)


def _candidate_installations(gams_dir: str | Path | None) -> list[GamsInstallation]:
    requested = gams_dir or os.getenv("PATHWAY_PILOT_GAMS_DIR")
    candidates: list[str | Path] = []
    if requested:
        candidates.append(requested)
    else:
        on_path = shutil.which("gams")
        if on_path:
            candidates.append(on_path)
        common_root = Path("C:/GAMS")
        if common_root.is_dir():
            candidates.extend(
                path.parent
                for path in sorted(
                    common_root.glob("*/gams.exe"),
                    key=_version_key,
                    reverse=True,
                )
            )

    installations: list[GamsInstallation] = []
    seen: set[Path] = set()
    for candidate in candidates:
        installation = _installation_from_path(candidate)
        if installation is None:
            continue
        resolved = installation.directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            installations.append(installation)
    return installations


def _probe_gams_cplex(installation: GamsInstallation) -> tuple[bool, str]:
    source = """option lp = cplex;
positive variable x;
variable z;
equations objective_definition, lower_bound;
objective_definition.. z =e= x;
lower_bound.. x =g= 1;
model license_probe / all /;
solve license_probe using lp minimizing z;
abort$(license_probe.solvestat <> 1 or license_probe.modelstat <> 1)
    'GAMS/CPLEX license probe failed', license_probe.solvestat, license_probe.modelstat;
"""
    temporary_root = Path(".tmp")
    temporary_root.mkdir(parents=True, exist_ok=True)
    source_path = temporary_root / "gams_license_probe.gms"
    listing_path = temporary_root / "gams_license_probe.lst"
    try:
        source_path.write_text(source, encoding="utf-8")
        completed = subprocess.run(
            [str(installation.gams), source_path.name, "lo=0"],
            cwd=temporary_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            check=False,
        )
    finally:
        source_path.unlink(missing_ok=True)
        listing_path.unlink(missing_ok=True)
    return completed.returncode == 0, completed.stdout


def find_gams_cplex(
    gams_dir: str | Path | None = None,
    *,
    verify_license: bool = True,
) -> GamsInstallation:
    """Find a GAMS installation whose CPLEX link can solve a licensed LP."""

    installations = _candidate_installations(gams_dir)
    if not installations:
        location = gams_dir or os.getenv("PATHWAY_PILOT_GAMS_DIR") or "PATH and C:\\GAMS"
        raise GamsCplexError(f"No complete GAMS installation found via {location}.")

    failures: list[str] = []
    for installation in installations:
        if not verify_license:
            return installation
        usable, output = _probe_gams_cplex(installation)
        if usable:
            return installation
        diagnostic = next(
            (
                line.strip()
                for line in output.splitlines()
                if "license" in line.lower() or "error" in line.lower()
            ),
            "license probe failed",
        )
        failures.append(f"{installation.directory}: {diagnostic}")

    raise GamsCplexError(
        "No installed GAMS version could provide a usable licensed CPLEX interface. "
        + "; ".join(failures)
    )


def _gams_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if "'" in value:
        raise GamsCplexError(f"GAMS bridge paths cannot contain a single quote: {value}")
    return value


def _augment_gams_program(program_path: Path, solution_path: Path) -> None:
    source = program_path.read_text(encoding="utf-8")
    solve_pattern = re.compile(r"solve m using rmip (minimizing|maximizing) obj;", re.IGNORECASE)
    match = solve_pattern.search(source)
    if match is None:
        raise GamsCplexError("mps2gms output did not contain the expected solve statement.")

    solve_statement = match.group(0)
    replacement = f"""scalar linopy_model_status, linopy_solve_status;
m.optfile = 1;
{solve_statement}
linopy_model_status = m.modelstat;
linopy_solve_status = m.solvestat;
execute_unload '{_gams_path(solution_path)}',
    xc, xb, xi, xsc, xsi, eg, el, ee, er, obj,
    linopy_model_status, linopy_solve_status;"""
    program_path.write_text(
        source[: match.start()] + replacement + source[match.end() :],
        encoding="utf-8",
    )


def _run_checked(command: list[str], *, cwd: Path, description: str) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise GamsCplexError(f"{description} failed with exit code {completed.returncode}:\n{tail}")
    return completed.stdout


def _dump_symbol(
    installation: GamsInstallation,
    solution_path: Path,
    symbol: str,
    work_dir: Path,
) -> pd.DataFrame:
    output_path = work_dir / f"{symbol}.csv"
    _run_checked(
        [
            str(installation.gdxdump),
            str(solution_path),
            f"symb={symbol}",
            f"output={output_path}",
            "format=csv",
            "CSVAllFields",
            "EpsOut=0",
        ],
        cwd=work_dir,
        description=f"gdxdump for {symbol}",
    )
    return pd.read_csv(output_path)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.replace({"EPS": 0.0, "+EPS": 0.0, "-EPS": 0.0}),
        errors="coerce",
    ).fillna(0.0)


def _scalar_value(frame: pd.DataFrame, symbol: str) -> float:
    if frame.empty or "Val" not in frame:
        raise GamsCplexError(f"GAMS solution does not contain scalar {symbol}.")
    return float(_numeric(frame["Val"]).iloc[0])


def _read_labelled_values(
    frames: list[pd.DataFrame],
    *,
    prefix: str,
    field: str,
    labels: np.ndarray,
) -> pd.Series:
    values = pd.Series(0.0, index=pd.Index(labels.astype(int)), dtype="float64")
    for frame in frames:
        if frame.empty:
            continue
        name_column = frame.columns[0]
        if field not in frame:
            raise GamsCplexError(f"GAMS solution column {field!r} is missing.")
        names = frame[name_column].astype(str)
        if not names.str.fullmatch(rf"{re.escape(prefix)}\d+").all():
            unexpected = names[~names.str.fullmatch(rf"{re.escape(prefix)}\d+")].iloc[0]
            raise GamsCplexError(f"Unexpected GAMS solution label: {unexpected!r}.")
        indices = names.str[len(prefix) :].astype(int)
        unknown = indices[~indices.isin(values.index)]
        if not unknown.empty:
            raise GamsCplexError(f"GAMS returned unknown {prefix}-label {unknown.iloc[0]}.")
        values.loc[indices.to_numpy()] = _numeric(frame[field]).to_numpy()
    return values


def _assign_solution(model: object, primal: pd.Series, dual: pd.Series, objective: float) -> None:
    model.objective.set_value(objective)
    model.status = "ok"
    model.termination_condition = "optimal"
    model.solver_model = None
    model.solver_name = "gams-cplex"

    for _, variable in model.variables.items():
        flat_labels = np.ravel(variable.labels)
        values = primal.reindex(flat_labels).to_numpy()
        variable.solution = xr.DataArray(values.reshape(variable.labels.shape), variable.coords)

    for _, constraint in model.constraints.items():
        flat_labels = np.ravel(constraint.labels)
        values = dual.reindex(flat_labels).to_numpy()
        constraint.dual = xr.DataArray(
            values.reshape(constraint.labels.shape),
            constraint.labels.coords,
        )


def solve_with_gams_cplex(
    model: object,
    *,
    gams_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    keep_files: bool = False,
    threads: int = 6,
    lp_method: int = 4,
) -> tuple[str, str]:
    """Solve a continuous Linopy LP through GAMS/CPLEX and assign its solution."""

    if model.type != "LP":
        raise GamsCplexError(
            f"The GAMS/CPLEX bridge currently supports continuous LPs, not {model.type}."
        )
    if threads < 1:
        raise ValueError("threads must be at least 1")
    if lp_method not in range(7):
        raise ValueError("lp_method must be a CPLEX LP method number from 0 through 6")

    installation = find_gams_cplex(gams_dir)
    print(f"Using licensed GAMS/CPLEX from {installation.directory.resolve()}")

    if work_dir is None:
        solver_dir = Path(".tmp/gams_cplex").resolve()
    else:
        solver_dir = Path(work_dir).resolve()
    solver_dir.mkdir(parents=True, exist_ok=True)

    managed_paths = [
        solver_dir / name
        for name in (
            "linopy_model.mps",
            "linopy_model.gdx",
            "linopy_model.gms",
            "linopy_model.lst",
            "linopy_solution.gdx",
            "gams_cplex.log",
            "cplex.opt",
            "xc.csv",
            "xb.csv",
            "xi.csv",
            "xsc.csv",
            "xsi.csv",
            "eg.csv",
            "el.csv",
            "ee.csv",
            "er.csv",
            "obj.csv",
            "linopy_model_status.csv",
            "linopy_solve_status.csv",
        )
    ]
    for path in managed_paths:
        path.unlink(missing_ok=True)

    try:
        model.matrices.clean_cached_properties()
        model.reset_solution()
        model.constraints.sanitize_zeros()
        model.constraints.sanitize_infinities()

        problem_path = solver_dir / "linopy_model.mps"
        data_path = solver_dir / "linopy_model.gdx"
        program_path = solver_dir / "linopy_model.gms"
        solution_path = solver_dir / "linopy_solution.gdx"
        log_path = solver_dir / "gams_cplex.log"

        model.to_file(problem_path, io_api="mps", explicit_coordinate_names=False)
        conversion_log = _run_checked(
            [
                str(installation.mps2gms),
                problem_path.name,
                data_path.name,
                program_path.name,
            ],
            cwd=solver_dir,
            description="mps2gms conversion",
        )
        _augment_gams_program(program_path, solution_path)
        (solver_dir / "cplex.opt").write_text(
            f"lpmethod {lp_method}\nthreads {threads}\nnames 0\n",
            encoding="utf-8",
        )
        solve_log = _run_checked(
            [
                str(installation.gams),
                program_path.name,
                "rmip=cplex",
                "lo=3",
                f"curDir={solver_dir}",
            ],
            cwd=solver_dir,
            description="GAMS/CPLEX solve",
        )
        log_path.write_text(conversion_log + "\n" + solve_log, encoding="utf-8")
        if not solution_path.is_file():
            raise GamsCplexError("GAMS completed without writing linopy_solution.gdx.")

        model_status = int(
            round(
                _scalar_value(
                    _dump_symbol(
                        installation, solution_path, "linopy_model_status", solver_dir
                    ),
                    "linopy_model_status",
                )
            )
        )
        solve_status = int(
            round(
                _scalar_value(
                    _dump_symbol(
                        installation, solution_path, "linopy_solve_status", solver_dir
                    ),
                    "linopy_solve_status",
                )
            )
        )
        if solve_status != 1 or model_status != 1:
            model.status = "warning"
            model.termination_condition = {
                3: "unbounded",
                4: "infeasible",
                5: "infeasible",
                6: "infeasible",
                18: "unbounded",
                19: "infeasible",
            }.get(model_status, "other")
            return model.status, model.termination_condition

        variable_frames = [
            _dump_symbol(installation, solution_path, symbol, solver_dir)
            for symbol in ("xc", "xb", "xi", "xsc", "xsi")
        ]
        constraint_frames = [
            _dump_symbol(installation, solution_path, symbol, solver_dir)
            for symbol in ("eg", "el", "ee", "er")
        ]
        primal = _read_labelled_values(
            variable_frames,
            prefix="x",
            field="Val",
            labels=np.asarray(model.matrices.vlabels),
        )
        dual = _read_labelled_values(
            constraint_frames,
            prefix="c",
            field="Marginal",
            labels=np.asarray(model.matrices.clabels),
        )
        objective = _scalar_value(
            _dump_symbol(installation, solution_path, "obj", solver_dir),
            "obj",
        )
        _assign_solution(model, primal, dual, objective)
        return "ok", "optimal"
    finally:
        if keep_files:
            print(f"Kept GAMS/CPLEX working files at {solver_dir}")
        else:
            for path in managed_paths:
                path.unlink(missing_ok=True)
