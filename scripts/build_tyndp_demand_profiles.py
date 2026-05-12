from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathway_pilot.config import DEV_DATA_DIR
from pathway_pilot.demand import build_demand_table


def main() -> None:
    input_dir = (
        DEV_DATA_DIR
        / "TYNDP24"
        / "Demand"
        / "Demand Profiles"
        / "NT"
        / "Electricity demand profiles"
    )
    output_dir = DEV_DATA_DIR / "pathway-pilot" / "tvar"
    output_path = output_dir / "tyndp_demand_profiles_2030_2040.parquet"

    table = build_demand_table(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)

    print(f"Wrote {len(table):,} rows to {output_path}")


if __name__ == "__main__":
    main()
