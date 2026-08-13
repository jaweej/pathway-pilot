"""Solver entry points for the pathway pilot."""

from __future__ import annotations

from pathlib import Path

import pypsa
from linopy_gams_cplex import solve as solve_with_cplex

from pathway_pilot.build_network import build_network
from pathway_pilot.model_config import ModelConfig
from pathway_pilot.model_inputs import ModelInputs


def solve_model(
    cfg: ModelConfig,
    data: ModelInputs,
    solver_name: str = "highs",
    gams_dir: str | Path | None = None,
    cplex_transfer: str = "direct",
    cplex_method: str = "barrier",
    cplex_threads: int = 6,
) -> tuple[pypsa.Network, str, str]:
    network = build_network(cfg, data)
    if solver_name == "gams-cplex":
        network._multi_invest = 1
        network._linearized_uc = False
        network.consistency_check(strict=["unknown_buses"])
        model = network.optimize.create_model(
            multi_investment_periods=True,
            consistency_check=False,
            include_objective_constant=False,
        )
        if cplex_transfer in {"direct", "gams"}:
            result = solve_with_cplex(
                model,
                gams_dir=gams_dir,
                method=cplex_method,
                threads=cplex_threads,
                transfer=cplex_transfer,
            )
            status = result.status
            condition = result.termination_condition
            network.meta["cplex_transfer"] = cplex_transfer
            network.meta["cplex_method"] = cplex_method
            network.meta["cplex_threads"] = cplex_threads
            if result.timings is not None:
                timings = result.timings
                network.meta["solver_timings_seconds"] = {
                    "model_export": timings.export_seconds,
                    "solver_read": timings.read_seconds,
                    "model_transfer": timings.export_seconds + timings.read_seconds,
                    "solver": timings.solve_seconds,
                }
                print(
                    "Direct CPLEX timings: "
                    f"MPS export {timings.export_seconds:.3f}s, "
                    f"CPLEX read {timings.read_seconds:.3f}s, "
                    f"CPLEX solve {timings.solve_seconds:.3f}s"
                )
        else:
            raise ValueError(
                f"Unknown CPLEX transfer mode {cplex_transfer!r}; expected 'direct' or 'gams'."
            )
        if status == "ok":
            network.optimize.assign_solution()
            network.optimize.assign_duals()
            network.optimize.post_processing()
        return network, status, condition

    status, condition = network.optimize(
        multi_investment_periods=True,
        solver_name=solver_name,
        include_objective_constant=False,
    )
    return network, status, condition
