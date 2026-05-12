import pandas as pd

from pathway_pilot.model_config import load_config
from pathway_pilot.technology_data import _gas_assumption, _renewable_assumption


SCHEMA_COLUMNS = [
    "year",
    "electrical_efficiency_net_annual_avg",
    "technical_lifetime_years",
    "total_nominal_investment_meur_per_mw_e",
    "variable_om_eur_per_mwh_e",
    "fixed_om_eur_per_mw_e_per_year",
]


def test_technology_assumptions_use_techcat_schema_and_gas_price_formula():
    cfg = load_config("config/model_config.yaml")
    wind = pd.DataFrame(
        [
            [2030, None, 30, 1.0, 2.0, 10_000],
            [2040, None, 30, 0.9, 2.0, 9_000],
            [2050, None, 30, 0.8, 2.0, 8_000],
        ],
        columns=SCHEMA_COLUMNS,
    )
    solar = pd.DataFrame(
        [
            [2030, None, 40, 0.5, None, 5_000],
            [2040, None, 40, 0.4, None, 4_000],
            [2050, None, 40, 0.3, None, 3_000],
        ],
        columns=SCHEMA_COLUMNS,
    )
    gas = pd.DataFrame(
        [
            [2030, 0.4, 25, 0.6, 4.0, 8_000],
            [2040, 0.5, 25, 0.5, 3.0, 7_000],
            [2050, 0.6, 25, 0.45, 2.0, 6_000],
        ],
        columns=SCHEMA_COLUMNS,
    )

    wind_assumption = _renewable_assumption(cfg, wind, cfg.investment_periods)
    solar_assumption = _renewable_assumption(cfg, solar, cfg.investment_periods)
    gas_assumption = _gas_assumption(cfg, gas, cfg.investment_periods)

    assert wind_assumption.lifetime_years == 30
    assert wind_assumption.unit_capex_by_period[2030] == 1_000_000
    assert wind_assumption.unit_capex_by_period[2050] == 800_000
    assert solar_assumption.unit_capex_by_period[2030] == 500_000
    assert solar_assumption.unit_capex_by_period[2050] == 300_000
    assert solar_assumption.marginal_cost_by_period[2030] == 0
    assert gas_assumption.marginal_cost_by_period[2030] == (80 * 3.6 / 7.47) / 0.4 + 4
    assert gas_assumption.unit_capex_by_period[2050] == 450_000
    assert gas_assumption.marginal_cost_by_period[2050] == (85 * 3.6 / 7.47) / 0.6 + 2
