from pathlib import Path

from pathway_pilot.model_config import load_config
from pathway_pilot.model_inputs import make_synthetic_inputs
from pathway_pilot.solve import solve_model


def test_smoke_solve_highs_tiny_fixture():
    cfg = load_config(Path("config/model_config.yaml"))
    data = make_synthetic_inputs(periods=cfg.investment_periods, hours_per_period=24)

    solved, status, condition = solve_model(cfg, data)

    assert status == "ok"
    assert condition == "optimal"
    assert solved.objective > 0
