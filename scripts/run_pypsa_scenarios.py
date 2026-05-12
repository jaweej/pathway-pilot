from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathway_pilot.config import DEV_DATA_DIR
from pathway_pilot.model_config import load_config, with_active_model

from run_pypsa_model import load_time_series, run_model_case, scenario_output_dir


def main() -> None:
    cfg = load_config(Path("config/model_config.yaml"))
    capacity_factors, demand = load_time_series()
    output_root = DEV_DATA_DIR / "pathway-pilot" / "output"
    climate_year_count = len(cfg.climate_years)

    for model_case in cfg.model_cases:
        case_cfg = with_active_model(cfg, model_case)
        for weather_year in cfg.climate_years:
            output_dir = scenario_output_dir(
                output_root,
                model_case,
                weather_year,
                climate_year_count=climate_year_count,
            )
            run_model_case(case_cfg, capacity_factors, demand, output_dir, weather_year)


if __name__ == "__main__":
    main()
