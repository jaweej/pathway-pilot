"""Direct Linopy LP solves via the CPLEX callable library bundled with GAMS."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from pathway_pilot.gams_cplex import GamsCplexError, GamsInstallation, find_gams_cplex


CPX_STAT_OPTIMAL = 1
CPX_STAT_UNBOUNDED = 2
CPX_STAT_INFEASIBLE = 3
CPX_STAT_INForUNBD = 4
CPX_PARAM_LPMETHOD = 1062
CPX_PARAM_THREADS = 1067
CPLEX_LP_METHODS = {
    "automatic": 0,
    "primal": 1,
    "dual": 2,
    "network": 3,
    "barrier": 4,
    "sifting": 5,
    "concurrent": 6,
}


@dataclass(frozen=True)
class CplexCallableTimings:
    """Wall-clock timings for the direct CPLEX bridge."""

    export_seconds: float
    read_seconds: float
    solve_seconds: float


class _CplexLibrary:
    def __init__(self, path: Path) -> None:
        try:
            self.dll = ctypes.WinDLL(str(path))
        except OSError as exc:
            raise GamsCplexError(f"Could not load the CPLEX callable library {path}: {exc}") from exc

        void_p = ctypes.c_void_p
        int_p = ctypes.POINTER(ctypes.c_int)
        double_p = ctypes.POINTER(ctypes.c_double)

        self._declare("CPXopenCPLEX", [int_p], void_p)
        self._declare("CPXcloseCPLEX", [ctypes.POINTER(void_p)], ctypes.c_int)
        self._declare("CPXcreateprob", [void_p, int_p, ctypes.c_char_p], void_p)
        self._declare("CPXfreeprob", [void_p, ctypes.POINTER(void_p)], ctypes.c_int)
        self._declare(
            "CPXreadcopyprob",
            [void_p, void_p, ctypes.c_char_p, ctypes.c_char_p],
            ctypes.c_int,
        )
        self._declare("CPXlpopt", [void_p, void_p], ctypes.c_int)
        self._declare("CPXgetstat", [void_p, void_p], ctypes.c_int)
        self._declare("CPXgetnumcols", [void_p, void_p], ctypes.c_int)
        self._declare("CPXgetnumrows", [void_p, void_p], ctypes.c_int)
        self._declare("CPXsetintparam", [void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int)
        self._declare(
            "CPXsolution",
            [void_p, void_p, int_p, double_p, double_p, double_p, double_p, double_p],
            ctypes.c_int,
        )
        self._declare(
            "CPXgeterrorstring",
            [void_p, ctypes.c_int, ctypes.c_char_p],
            ctypes.c_char_p,
        )

    def _declare(self, name: str, argtypes: list[object], restype: object) -> None:
        try:
            function = getattr(self.dll, name)
        except AttributeError as exc:
            raise GamsCplexError(
                f"The bundled CPLEX library does not export required function {name}."
            ) from exc
        function.argtypes = argtypes
        function.restype = restype

    def error_message(self, environment: int | None, code: int) -> str:
        buffer = ctypes.create_string_buffer(4096)
        self.dll.CPXgeterrorstring(environment, code, buffer)
        decoded = buffer.value.decode(errors="replace").strip()
        return decoded or f"CPLEX error {code}"


def _assign_solution(model: object, primal: np.ndarray, dual: np.ndarray, objective: float) -> None:
    if primal.size != len(model.matrices.vlabels):
        raise GamsCplexError(
            f"CPLEX returned {primal.size} primal values for "
            f"{len(model.matrices.vlabels)} Linopy variables."
        )
    if dual.size != len(model.matrices.clabels):
        raise GamsCplexError(
            f"CPLEX returned {dual.size} dual values for "
            f"{len(model.matrices.clabels)} Linopy constraints."
        )
    model.status = "ok"
    model.termination_condition = "optimal"
    model.solver_model = None
    model.solver_name = "gams-cplex"

    primal_by_label = pd.Series(primal, index=model.matrices.vlabels, dtype="float64")
    variables = model.variables.flat
    fixed = variables.loc[variables["lower"] == variables["upper"]]
    if not fixed.empty:
        # Fixed-format MPS can round large bounds materially (PyPSA's fixed
        # objective_constant is a common example). These are not optimizer
        # decisions, so restore their exact source-model values before mapping
        # the solution and evaluating the original objective expression.
        exact_fixed = fixed.set_index("labels")["lower"].astype("float64")
        primal_by_label.update(exact_fixed)
    dual_by_label = pd.Series(dual, index=model.matrices.clabels, dtype="float64")
    for _, variable in model.variables.items():
        labels = np.ravel(variable.labels)
        values = primal_by_label.reindex(labels, fill_value=np.nan).to_numpy()
        variable.solution = xr.DataArray(values.reshape(variable.labels.shape), variable.coords)
    for _, constraint in model.constraints.items():
        labels = np.ravel(constraint.labels)
        values = dual_by_label.reindex(labels, fill_value=np.nan).to_numpy()
        constraint.dual = xr.DataArray(
            values.reshape(constraint.labels.shape),
            constraint.labels.coords,
        )
    # MPS is a text format and rounds coefficients on export. Re-evaluate the
    # original Linopy expression with CPLEX's returned primal values so the
    # reported objective retains the source model's full-precision coefficients.
    # The raw CPLEX value remains useful internally for detecting API failures,
    # but is not the most accurate objective for the original in-memory model.
    model.objective.set_value(float(model.objective.expression.solution))


def _library_path(installation: GamsInstallation) -> Path:
    candidates = sorted(
        (
            path
            for path in installation.directory.glob("cplex*.dll")
            if path.stem.lower().removeprefix("cplex").isdigit()
        ),
        reverse=True,
    )
    if not candidates:
        raise GamsCplexError(
            f"No CPLEX callable library was found under {installation.directory}."
        )
    return candidates[0]


def solve_with_cplex_callable(
    model: object,
    *,
    gams_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    keep_files: bool = False,
    threads: int = 6,
    method: str = "barrier",
) -> tuple[str, str, CplexCallableTimings]:
    """Solve a continuous Linopy LP directly with GAMS-bundled CPLEX."""

    if model.type != "LP":
        raise GamsCplexError(
            f"The direct CPLEX bridge currently supports continuous LPs, not {model.type}."
        )
    if threads < 1:
        raise ValueError("threads must be at least 1")
    if method not in CPLEX_LP_METHODS:
        choices = ", ".join(CPLEX_LP_METHODS)
        raise ValueError(f"Unknown CPLEX LP method {method!r}; expected one of: {choices}.")

    # Select an installation that passes an actual licensed GAMS/CPLEX solve.
    # This deliberately excludes newer installations whose binaries are present
    # but whose GAMS license maintenance date is incompatible.
    installation = find_gams_cplex(gams_dir)
    library_path = _library_path(installation)
    library = _CplexLibrary(library_path)
    print(f"Using direct CPLEX callable library from {library_path.resolve()}")

    solver_dir = Path(work_dir or ".tmp/gams_cplex").resolve()
    solver_dir.mkdir(parents=True, exist_ok=True)
    problem_path = solver_dir / "linopy_model.mps"
    problem_path.unlink(missing_ok=True)

    model.matrices.clean_cached_properties()
    model.reset_solution()
    model.constraints.sanitize_zeros()
    model.constraints.sanitize_infinities()

    export_start = perf_counter()
    model.to_file(problem_path, io_api="mps", explicit_coordinate_names=False)
    export_seconds = perf_counter() - export_start

    environment: int | None = None
    problem: int | None = None
    try:
        status = ctypes.c_int()
        environment = library.dll.CPXopenCPLEX(ctypes.byref(status))
        if not environment or status.value:
            raise GamsCplexError(
                f"CPXopenCPLEX failed: {library.error_message(environment, status.value)}"
            )

        for parameter, value in (
            (CPX_PARAM_LPMETHOD, CPLEX_LP_METHODS[method]),
            (CPX_PARAM_THREADS, threads),
        ):
            result = library.dll.CPXsetintparam(environment, parameter, value)
            if result:
                raise GamsCplexError(
                    f"CPXsetintparam({parameter}) failed: "
                    f"{library.error_message(environment, result)}"
                )

        problem = library.dll.CPXcreateprob(
            environment,
            ctypes.byref(status),
            b"pathway_pilot",
        )
        if not problem or status.value:
            raise GamsCplexError(
                f"CPXcreateprob failed: {library.error_message(environment, status.value)}"
            )

        read_start = perf_counter()
        result = library.dll.CPXreadcopyprob(
            environment,
            problem,
            str(problem_path).encode(),
            None,
        )
        read_seconds = perf_counter() - read_start
        if result:
            raise GamsCplexError(
                f"CPXreadcopyprob failed: {library.error_message(environment, result)}"
            )

        n_columns = library.dll.CPXgetnumcols(environment, problem)
        n_rows = library.dll.CPXgetnumrows(environment, problem)
        if n_columns != len(model.matrices.vlabels) or n_rows != len(model.matrices.clabels):
            raise GamsCplexError(
                "CPLEX read a different model shape: "
                f"{n_rows} rows/{n_columns} columns instead of "
                f"{len(model.matrices.clabels)} rows/{len(model.matrices.vlabels)} columns."
            )

        solve_start = perf_counter()
        result = library.dll.CPXlpopt(environment, problem)
        solve_seconds = perf_counter() - solve_start
        if result:
            raise GamsCplexError(
                f"CPXlpopt failed: {library.error_message(environment, result)}"
            )

        solution_status = library.dll.CPXgetstat(environment, problem)
        condition = {
            CPX_STAT_OPTIMAL: "optimal",
            CPX_STAT_UNBOUNDED: "unbounded",
            CPX_STAT_INFEASIBLE: "infeasible",
            CPX_STAT_INForUNBD: "infeasible_or_unbounded",
        }.get(solution_status, "other")
        if condition != "optimal":
            model.status = "warning"
            model.termination_condition = condition
            return (
                model.status,
                condition,
                CplexCallableTimings(export_seconds, read_seconds, solve_seconds),
            )

        primal_buffer = (ctypes.c_double * n_columns)()
        dual_buffer = (ctypes.c_double * n_rows)()
        slack_buffer = (ctypes.c_double * n_rows)()
        reduced_cost_buffer = (ctypes.c_double * n_columns)()
        lp_status = ctypes.c_int()
        objective = ctypes.c_double()
        result = library.dll.CPXsolution(
            environment,
            problem,
            ctypes.byref(lp_status),
            ctypes.byref(objective),
            primal_buffer,
            dual_buffer,
            slack_buffer,
            reduced_cost_buffer,
        )
        if result:
            raise GamsCplexError(
                f"CPXsolution failed: {library.error_message(environment, result)}"
            )
        primal = np.ctypeslib.as_array(primal_buffer).copy()
        dual = np.ctypeslib.as_array(dual_buffer).copy()
        _assign_solution(model, primal, dual, objective.value)
        return (
            "ok",
            "optimal",
            CplexCallableTimings(export_seconds, read_seconds, solve_seconds),
        )
    finally:
        if problem and environment:
            problem_handle = ctypes.c_void_p(problem)
            library.dll.CPXfreeprob(environment, ctypes.byref(problem_handle))
        if environment:
            environment_handle = ctypes.c_void_p(environment)
            library.dll.CPXcloseCPLEX(ctypes.byref(environment_handle))
        if keep_files:
            print(f"Kept direct CPLEX problem file at {problem_path}")
        else:
            problem_path.unlink(missing_ok=True)
