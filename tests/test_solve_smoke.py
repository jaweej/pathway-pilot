from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathway_pilot.model_config import load_config, with_active_model
from pathway_pilot.model_inputs import ModelInputs, make_synthetic_inputs
from pathway_pilot.solve import solve_model


def test_smoke_solve_highs_tiny_fixture():
    cfg = load_config(Path("config/model_config.yaml"))
    data = make_synthetic_inputs(periods=cfg.investment_periods, hours_per_period=24)

    solved, status, condition = solve_model(cfg, data)

    assert status == "ok"
    assert condition == "optimal"
    assert solved.objective > 0


def test_smoke_solve_combined_dk_nl_tiny_fixture():
    cfg = with_active_model(load_config(Path("config/model_config.yaml")), "DK_NL")
    base = make_synthetic_inputs(periods=cfg.investment_periods, hours_per_period=4)
    demand_by_bus = pd.DataFrame(
        {
            "DK": base.demand_series * 0.35,
            "NL": base.demand_series * 0.65,
        }
    )
    wind_cf_by_bus = pd.DataFrame(
        {
            "DK": base.wind_cf_series,
            "NL": (base.wind_cf_series * 0.9).clip(upper=1),
        }
    )
    solar_cf_by_bus = pd.DataFrame(
        {
            "DK": base.solar_cf_series,
            "NL": (base.solar_cf_series * 1.1).clip(upper=1),
        }
    )
    data = ModelInputs(
        demand_series=demand_by_bus.sum(axis=1),
        wind_cf_series=wind_cf_by_bus.mean(axis=1),
        solar_cf_series=solar_cf_by_bus.mean(axis=1),
        demand_by_bus=demand_by_bus,
        wind_cf_by_bus=wind_cf_by_bus,
        solar_cf_by_bus=solar_cf_by_bus,
    )

    solved, status, condition = solve_model(cfg, data)

    assert status == "ok"
    assert condition == "optimal"
    assert "DK_NL_interconnector" in solved.links.index
    assert solved.links.loc["DK_NL_interconnector", "p_nom"] == 1000


@pytest.mark.skipif(
    not Path("C:/GAMS/37/gams.exe").is_file(),
    reason="Licensed GAMS 37 installation is not available",
)
def test_gams_cplex_matches_highs_tiny_fixture():
    cfg = with_active_model(load_config(Path("config/model_config.yaml")), "DK")
    data = make_synthetic_inputs(periods=cfg.investment_periods, hours_per_period=6)

    highs, highs_status, highs_condition = solve_model(cfg, data, solver_name="highs")
    cplex, cplex_status, cplex_condition = solve_model(
        cfg,
        data,
        solver_name="gams-cplex",
        gams_dir=Path("C:/GAMS/37"),
    )

    assert (highs_status, highs_condition) == ("ok", "optimal")
    assert (cplex_status, cplex_condition) == ("ok", "optimal")
    assert cplex.meta["cplex_transfer"] == "direct"
    assert cplex.meta["cplex_method"] == "barrier"
    assert cplex.meta["cplex_threads"] == 6
    assert cplex.meta["solver_timings_seconds"]["model_transfer"] >= 0
    assert cplex.model.solver_name == "gams-cplex"
    assert cplex.objective == pytest.approx(highs.objective, rel=1e-8)
    np.testing.assert_allclose(
        cplex.generators["p_nom_opt"],
        highs.generators["p_nom_opt"],
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        cplex.generators_t.p,
        highs.generators_t.p,
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        cplex.buses_t.marginal_price,
        highs.buses_t.marginal_price,
        rtol=1e-6,
        atol=1e-5,
    )
