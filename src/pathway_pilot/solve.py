"""Solver entry points for the pathway pilot."""

from __future__ import annotations

import pypsa

from pathway_pilot.build_network import build_network
from pathway_pilot.model_config import ModelConfig
from pathway_pilot.model_inputs import ModelInputs


def solve_model(
    cfg: ModelConfig,
    data: ModelInputs,
    solver_name: str = "highs",
) -> tuple[pypsa.Network, str, str]:
    network = build_network(cfg, data)
    status, condition = network.optimize(
        multi_investment_periods=True,
        solver_name=solver_name,
        include_objective_constant=False,
    )
    return network, status, condition
