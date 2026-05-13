from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pathway_pilot.config import DEV_DATA_DIR
from pathway_pilot.model_config import load_config


OUTPUT_DIR = DEV_DATA_DIR / "pathway-pilot" / "output"
DASHBOARD_PATH = OUTPUT_DIR / "pypsa_output_dashboard.html"
CAPACITY_CARRIERS = ["gas_turbine_cc", "gas", "wind", "solar"]
DISPATCH_CARRIERS = [*CAPACITY_CARRIERS, "load_shedding"]
GAS_CARRIERS = ["gas", "gas_turbine_cc"]
CONFIG_PATH = Path("config/model_config.yaml")


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _series_points(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    records = frame[columns].to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, pd.Timestamp):
                record[key] = value.strftime("%Y-%m-%d %H:%M")
            elif isinstance(value, float):
                record[key] = _round(value)
    return records


def _region_from_generator(generator: str, regions: set[str]) -> str | None:
    for region in regions:
        if generator.startswith(f"{region}_"):
            return region
    return None


def _regional_dispatch(dispatch: pd.DataFrame, region: str, regions: set[str]) -> pd.DataFrame:
    if "bus" in dispatch.columns:
        return dispatch[dispatch["bus"] == region].copy()
    region_by_generator = {
        generator: _region_from_generator(str(generator), regions)
        for generator in dispatch["generator"].unique()
    }
    return dispatch[dispatch["generator"].map(region_by_generator) == region].copy()


def _regional_capacities(capacities: pd.DataFrame, region: str, regions: set[str]) -> pd.DataFrame:
    if "bus" in capacities.columns:
        return capacities[capacities["bus"] == region].copy()
    region_by_generator = {
        generator: _region_from_generator(str(generator), regions)
        for generator in capacities["generator"].unique()
    }
    return capacities[capacities["generator"].map(region_by_generator) == region].copy()


def _regional_imports(flows: pd.DataFrame | None, region: str) -> pd.DataFrame | None:
    if flows is None or flows.empty:
        return None
    rows = []
    for flow in flows.itertuples(index=False):
        if flow.bus0 == region:
            import_mw = -float(flow.flow_bus0_to_bus1_mw)
        elif flow.bus1 == region:
            import_mw = float(flow.flow_bus0_to_bus1_mw)
        else:
            continue
        rows.append(
            {
                "period": flow.period,
                "timestep": flow.timestep,
                "import_mw": import_mw,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).groupby(["period", "timestep"], as_index=False)["import_mw"].sum()


WEEK_COLUMNS = [
    "timestep",
    "demand",
    "wind",
    "solar",
    "gas",
    "gas_turbine_cc",
    "load_shedding",
    "interconnector_import",
    "interconnector_export",
    "price_eur_per_mwh",
]


def _week_window(rows: pd.DataFrame, start_index: int, window_size: int = 168) -> pd.DataFrame:
    if rows.empty:
        return rows
    start_index = max(0, min(start_index, max(len(rows) - window_size, 0)))
    return rows.iloc[start_index : start_index + window_size]


def _fixed_week(rows: pd.DataFrame, month_day: str) -> pd.DataFrame:
    if rows.empty:
        return rows
    year = rows.iloc[0]["timestep"].year
    target = pd.Timestamp(f"{year}-{month_day}")
    matches = rows.index[rows["timestep"] >= target]
    start_index = int(matches[0]) if len(matches) else max(len(rows) - 168, 0)
    return _week_window(rows, start_index)


def _highest_shedding_week(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    window_size = 168
    if len(rows) <= window_size:
        return rows
    shed = rows["load_shedding"].clip(lower=0)
    positive = shed > 1e-6
    if not positive.any():
        return _week_window(rows, 0)
    groups = positive.ne(positive.shift()).cumsum()
    best_episode = None
    best_energy = 0.0
    for _, episode in shed[positive].groupby(groups[positive]):
        energy = float(episode.sum())
        if energy > best_energy:
            best_energy = energy
            best_episode = episode
    if best_episode is None or best_energy <= 1e-6:
        return _week_window(rows, 0)
    episode_center = float((best_episode.index.to_series() * best_episode).sum() / best_energy)
    start_index = round(episode_center - (window_size - 1) / 2)
    return _week_window(rows, start_index)


def _week_label(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""
    start = rows.iloc[0]["timestep"].strftime("%Y-%m-%d")
    end = rows.iloc[-1]["timestep"].strftime("%Y-%m-%d")
    return f"{start} to {end}"


def _build_week_data(hourly: pd.DataFrame, compact_week_data: bool) -> tuple[dict, str | None, str | None]:
    week_data = {}
    date_min = None
    date_max = None
    for period, group in hourly.groupby("period"):
        ordered = group.sort_values("timestep").reset_index(drop=True)
        if compact_week_data:
            windows = {
                "winter": _fixed_week(ordered, "01-15"),
                "summer": _fixed_week(ordered, "07-15"),
                "shed": _highest_shedding_week(ordered),
            }
            week_data[str(period)] = {
                key: {
                    "label": _week_label(window),
                    "rows": _series_points(window, WEEK_COLUMNS),
                }
                for key, window in windows.items()
            }
        else:
            week_data[str(period)] = _series_points(ordered, WEEK_COLUMNS)
            if date_min is None:
                date_min = ordered.iloc[0]["timestep"].strftime("%Y-%m-%d")
            date_max = ordered.iloc[-1]["timestep"].strftime("%Y-%m-%d")
    return week_data, date_min, date_max


def _dashboard_payload(
    capacities: pd.DataFrame,
    dispatch: pd.DataFrame,
    prices: pd.DataFrame,
    metadata: dict,
    region: str | None = None,
    flows: pd.DataFrame | None = None,
    compact_week_data: bool = True,
) -> dict:
    regions = set((metadata.get("model_regions") or {}).keys())
    interconnector_capacity_mw = sum(
        float(interconnector.get("capacity_mw", 0.0))
        for interconnector in metadata.get("interconnectors", [])
        if region is not None
        and region in {interconnector.get("bus0"), interconnector.get("bus1")}
    )
    if region is not None:
        capacities = _regional_capacities(capacities, region, regions)
        dispatch = _regional_dispatch(dispatch, region, regions)

    dispatch_by_carrier = (
        dispatch.groupby(["period", "timestep", "carrier"], as_index=False)["dispatch_mw"].sum()
    )
    dispatch_wide = (
        dispatch_by_carrier.pivot(index=["period", "timestep"], columns="carrier", values="dispatch_mw")
        .fillna(0)
        .reset_index()
    )
    for carrier in DISPATCH_CARRIERS:
        if carrier not in dispatch_wide:
            dispatch_wide[carrier] = 0.0

    imports = _regional_imports(flows, region) if region is not None else None
    if imports is not None:
        dispatch_wide = dispatch_wide.merge(imports, on=["period", "timestep"], how="left")
        dispatch_wide["import_mw"] = dispatch_wide["import_mw"].fillna(0.0)
    else:
        dispatch_wide["import_mw"] = 0.0
    dispatch_wide["interconnector_import"] = dispatch_wide["import_mw"].clip(lower=0)
    dispatch_wide["interconnector_export"] = dispatch_wide["import_mw"].clip(upper=0)
    dispatch_wide["demand"] = dispatch_wide[DISPATCH_CARRIERS].sum(axis=1) + dispatch_wide[
        "import_mw"
    ]

    preferred_bus = region or ("electricity" if "electricity" in set(prices["bus"]) else prices["bus"].iloc[0])
    prices = prices[prices["bus"] == preferred_bus].drop(columns=["bus"])
    hourly = dispatch_wide.merge(prices, on=["period", "timestep"], how="left")
    dispatch_carriers = [*CAPACITY_CARRIERS]
    if region is not None and imports is not None:
        dispatch_carriers.extend(["interconnector_import", "interconnector_export"])
    dispatch_carriers.append("load_shedding")

    capacity_rows = capacities.copy()
    capacity_rows["p_nom_opt_mw"] = capacity_rows["p_nom_opt_mw"].clip(lower=0)
    gas_marginal_cost_max = float(
        capacity_rows.loc[capacity_rows["carrier"].isin(GAS_CARRIERS), "marginal_cost"].max()
    )

    build_capacity = (
        capacity_rows.groupby(["build_year", "carrier"], as_index=False)["p_nom_opt_mw"].sum()
    )
    build_capacity = build_capacity[build_capacity["carrier"] != "load_shedding"]

    capex_rows = capacity_rows[capacity_rows["carrier"] != "load_shedding"].copy()
    if "unit_capex_eur_per_mw" not in capex_rows.columns:
        capex_rows["unit_capex_eur_per_mw"] = capex_rows["capital_cost"]
    capex_rows["capex_meur_per_mw"] = capex_rows["unit_capex_eur_per_mw"] / 1_000_000
    build_capex = capex_rows[
        ["build_year", "carrier", "capex_meur_per_mw"]
    ].copy()

    active_rows = []
    for period in sorted(hourly["period"].unique()):
        active = capacity_rows[
            (capacity_rows["build_year"] <= period)
            & (period < capacity_rows["build_year"] + capacity_rows["lifetime"])
            & (capacity_rows["carrier"] != "load_shedding")
        ]
        grouped = active.groupby("carrier", as_index=False)["p_nom_opt_mw"].sum()
        for row in grouped.itertuples(index=False):
            active_rows.append(
                {
                    "period": int(period),
                    "carrier": row.carrier,
                    "p_nom_opt_mw": _round(row.p_nom_opt_mw),
                }
            )

    duration = {}
    for period, group in hourly.groupby("period"):
        sorted_group = group.sort_values("price_eur_per_mwh", ascending=False).reset_index(drop=True)
        sorted_group["hour"] = sorted_group.index + 1
        duration[str(period)] = _series_points(
            sorted_group,
            ["hour", "price_eur_per_mwh", "demand"],
        )

    capture = []
    for period, group in hourly.groupby("period"):
        time_price = group["price_eur_per_mwh"].mean()
        demand_price = (group["demand"] * group["price_eur_per_mwh"]).sum() / group["demand"].sum()
        capture.append(
            {
                "period": int(period),
                "technology": "demand",
                "capture_price": _round(demand_price),
                "reference_price": _round(time_price),
                "capture_rate": _round(demand_price / time_price),
            }
        )
        for technology in CAPACITY_CARRIERS:
            generation = group[technology]
            capture_price = (
                (generation * group["price_eur_per_mwh"]).sum() / generation.sum()
                if generation.sum() > 0
                else 0
            )
            capture.append(
                {
                    "period": int(period),
                    "technology": technology,
                    "capture_price": _round(capture_price),
                    "reference_price": _round(demand_price),
                    "capture_rate": _round(capture_price / demand_price),
                }
            )

    week_data, date_min, date_max = _build_week_data(hourly, compact_week_data)

    summary = []
    security = []
    generation_shares = []
    for period, group in hourly.groupby("period"):
        shed = group["load_shedding"].clip(lower=0)
        shed_hours = int((shed > 1e-6).sum())
        shed_energy_mwh = float(shed.sum())
        demand_energy_mwh = float(group["demand"].sum())
        import_energy_mwh = float(group["interconnector_import"].clip(lower=0).sum())
        export_energy_mwh = float((-group["interconnector_export"].clip(upper=0)).sum())
        total_interconnector_capacity_mwh = interconnector_capacity_mw * len(group)
        generation_totals = {
            "gas_turbine_cc": float(group["gas_turbine_cc"].clip(lower=0).sum()),
            "gas": float(group["gas"].clip(lower=0).sum()),
            "wind": float(group["wind"].clip(lower=0).sum()),
            "solar": float(group["solar"].clip(lower=0).sum()),
            "load_shedding": shed_energy_mwh,
        }
        generation_total = sum(generation_totals.values())
        for carrier, value in generation_totals.items():
            generation_shares.append(
                {
                    "period": int(period),
                    "carrier": carrier,
                    "share": _round(100 * value / generation_total if generation_total else 0),
                    "energy_twh": _round(value / 1_000_000),
                }
            )
        summary.append(
            {
                "period": int(period),
                "peak_demand_mw": _round(group["demand"].max()),
                "energy_twh": _round(demand_energy_mwh / 1_000_000),
                "average_price": _round(group["price_eur_per_mwh"].mean(), 2),
                "load_shedding_mwh": _round(shed_energy_mwh, 4),
            }
        )
        security.append(
            {
                "period": int(period),
                "lole_hours": shed_hours,
                "eens_mwh": _round(shed_energy_mwh, 4),
                "eens_gwh": _round(shed_energy_mwh / 1_000, 6),
                "eens_pct_demand": _round(
                    100 * shed_energy_mwh / demand_energy_mwh if demand_energy_mwh else 0,
                    6,
                ),
                "peak_shed_mw": _round(shed.max(), 4),
                "gas_generation_twh": _round(group[GAS_CARRIERS].sum(axis=1).sum() / 1_000_000, 3),
                "import_twh": _round(import_energy_mwh / 1_000_000, 3),
                "export_twh": _round(export_energy_mwh / 1_000_000, 3),
                "interconnector_utilisation_pct": _round(
                    100
                    * (import_energy_mwh + export_energy_mwh)
                    / total_interconnector_capacity_mwh
                    if total_interconnector_capacity_mwh
                    else 0,
                    3,
                ),
                "average_price_eur_per_mwh": _round(group["price_eur_per_mwh"].mean(), 3),
                "average_price_without_shedding_eur_per_mwh": _round(
                    group.loc[shed <= 1e-6, "price_eur_per_mwh"].mean()
                    if (shed <= 1e-6).any()
                    else 0,
                    3,
                ),
                "voll_eur_per_mwh": _round(
                    capacities.loc[
                        capacities["carrier"] == "load_shedding", "marginal_cost"
                    ].iloc[0],
                    2,
                ),
            }
        )

    return {
        "summary": summary,
        "averagePrices": [
            {
                "period": row["period"],
                "average_price_eur_per_mwh": row["average_price"],
            }
            for row in summary
        ],
        "buildCapacity": _series_points(build_capacity, ["build_year", "carrier", "p_nom_opt_mw"]),
        "buildCapex": _series_points(build_capex, ["build_year", "carrier", "capex_meur_per_mw"]),
        "activeCapacity": active_rows,
        "duration": duration,
        "capture": capture,
        "generationShares": generation_shares,
        "security": security,
        "gasMarginalCostMax": _round(gas_marginal_cost_max),
        "capacityCarriers": CAPACITY_CARRIERS,
        "captureTechnologies": [*CAPACITY_CARRIERS, "demand"],
        "dispatchCarriers": dispatch_carriers,
        "weekData": week_data,
        "weekDataMode": "compact" if compact_week_data else "full",
        "dateMin": date_min,
        "dateMax": date_max,
        "metadata": {**metadata, "selected_region": region},
    }


def load_dashboard_data(output_dir: Path = OUTPUT_DIR, compact_week_data: bool = True) -> dict:
    capacities = pd.read_parquet(output_dir / "optimal_capacities.parquet")
    dispatch = pd.read_parquet(output_dir / "hourly_dispatch.parquet")
    prices = pd.read_parquet(output_dir / "hourly_prices.parquet")
    flows_path = output_dir / "hourly_interconnector_flows.parquet"
    flows = pd.read_parquet(flows_path) if flows_path.exists() else None
    metadata_path = output_dir / "model_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {
            "active_model": "DK",
            "demand_zones": ["DKE1", "DKW1"],
            "capacity_factor_zone": "DKW1",
            "weather_year": 1982,
        }
    )
    data = _dashboard_payload(
        capacities,
        dispatch,
        prices,
        metadata,
        flows=flows,
        compact_week_data=compact_week_data,
    )
    model_regions = metadata.get("model_regions") or {}
    if len(model_regions) > 1:
        data["regions"] = {
            region: _dashboard_payload(
                capacities,
                dispatch,
                prices,
                metadata,
                region=region,
                flows=flows,
                compact_week_data=compact_week_data,
            )
            for region in model_regions
        }
        data["defaultRegion"] = next(iter(model_regions))
    return data


def has_output_tables(path: Path) -> bool:
    return all(
        (path / file_name).exists()
        for file_name in [
            "optimal_capacities.parquet",
            "hourly_dispatch.parquet",
            "hourly_prices.parquet",
        ]
    )


def _load_period_weights() -> dict[int, float]:
    try:
        cfg = load_config(CONFIG_PATH)
    except (FileNotFoundError, KeyError, ValueError):
        return {}
    return {int(period): float(weight) for period, weight in cfg.period_weights.items()}


def _period_weight(period: int, period_weights: dict[int, float]) -> float:
    return float(period_weights.get(int(period), 1.0))


def _scenario_frames(scenario_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    capacities = pd.read_parquet(scenario_dir / "optimal_capacities.parquet")
    dispatch = pd.read_parquet(scenario_dir / "hourly_dispatch.parquet")
    prices = pd.read_parquet(scenario_dir / "hourly_prices.parquet")
    flows_path = scenario_dir / "hourly_interconnector_flows.parquet"
    flows = pd.read_parquet(flows_path) if flows_path.exists() else pd.DataFrame()
    metadata_path = scenario_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return capacities, dispatch, prices, flows, metadata


def _regional_demand_from_dispatch(
    dispatch: pd.DataFrame,
    flows: pd.DataFrame,
    region: str,
    regions: set[str],
) -> pd.DataFrame:
    regional_dispatch = _regional_dispatch(dispatch, region, regions)
    generation = regional_dispatch.groupby(["period", "timestep"], as_index=False)[
        "dispatch_mw"
    ].sum()
    generation = generation.rename(columns={"dispatch_mw": "generation_mw"})
    imports = _regional_imports(flows, region)
    if imports is None:
        generation["import_mw"] = 0.0
    else:
        generation = generation.merge(imports, on=["period", "timestep"], how="left")
        generation["import_mw"] = generation["import_mw"].fillna(0.0)
    generation["demand_mw"] = generation["generation_mw"] + generation["import_mw"]
    return generation[["period", "timestep", "demand_mw"]]


def _consumer_surplus_proxy(
    dispatch: pd.DataFrame,
    prices: pd.DataFrame,
    flows: pd.DataFrame,
    region: str,
    regions: set[str],
    period_weights: dict[int, float],
) -> dict[int, float]:
    demand = _regional_demand_from_dispatch(dispatch, flows, region, regions)
    regional_prices = prices[prices["bus"] == region][
        ["period", "timestep", "price_eur_per_mwh"]
    ]
    frame = demand.merge(regional_prices, on=["period", "timestep"], how="left")
    frame["weight"] = frame["period"].map(lambda period: _period_weight(period, period_weights))
    frame["value"] = -(frame["demand_mw"] * frame["price_eur_per_mwh"] * frame["weight"])
    return {int(period): float(group["value"].sum()) for period, group in frame.groupby("period")}


def _producer_surplus(
    capacities: pd.DataFrame,
    dispatch: pd.DataFrame,
    prices: pd.DataFrame,
    region: str,
    regions: set[str],
    period_weights: dict[int, float],
) -> dict[int, float]:
    regional_dispatch = _regional_dispatch(dispatch, region, regions)
    generator_costs = capacities[["generator", "marginal_cost"]]
    regional_dispatch = regional_dispatch.merge(generator_costs, on="generator", how="left")
    regional_prices = prices[prices["bus"] == region][
        ["period", "timestep", "price_eur_per_mwh"]
    ]
    frame = regional_dispatch.merge(regional_prices, on=["period", "timestep"], how="left")
    frame["weight"] = frame["period"].map(lambda period: _period_weight(period, period_weights))
    frame["revenue"] = frame["dispatch_mw"] * frame["price_eur_per_mwh"] * frame["weight"]
    frame["variable_cost"] = frame["dispatch_mw"] * frame["marginal_cost"] * frame["weight"]
    values = (frame["revenue"] - frame["variable_cost"]).groupby(frame["period"]).sum().to_dict()

    regional_capacities = _regional_capacities(capacities, region, regions)
    for row in regional_capacities.itertuples(index=False):
        if row.carrier == "load_shedding":
            continue
        for period, weight in period_weights.items():
            if row.build_year <= period < row.build_year + row.lifetime:
                values[period] = values.get(period, 0.0) - (
                    float(row.p_nom_opt_mw) * float(row.capital_cost) * weight
                )

    return {int(period): float(value) for period, value in values.items()}


def _congestion_rents(
    flows: pd.DataFrame,
    prices: pd.DataFrame,
    regions: set[str],
    period_weights: dict[int, float],
) -> dict[str, dict[int, float]]:
    rents = {region: {} for region in regions}
    if flows.empty:
        return rents
    price_wide = prices.pivot(index=["period", "timestep"], columns="bus", values="price_eur_per_mwh")
    frame = flows.merge(price_wide.reset_index(), on=["period", "timestep"], how="left")
    for row in frame.itertuples(index=False):
        if row.bus0 not in regions or row.bus1 not in regions:
            continue
        price0 = float(getattr(row, row.bus0))
        price1 = float(getattr(row, row.bus1))
        weight = _period_weight(row.period, period_weights)
        rent = (price1 - price0) * float(row.flow_bus0_to_bus1_mw) * weight
        period = int(row.period)
        rents[row.bus0][period] = rents[row.bus0].get(period, 0.0) + rent / 2
        rents[row.bus1][period] = rents[row.bus1].get(period, 0.0) + rent / 2
    return rents


def _welfare_components(scenario_dir: Path, period_weights: dict[int, float]) -> dict[str, dict[int, dict[str, float]]]:
    capacities, dispatch, prices, flows, metadata = _scenario_frames(scenario_dir)
    regions = set((metadata.get("model_regions") or {}).keys())
    congestion = _congestion_rents(flows, prices, regions, period_weights)
    components = {}
    for region in sorted(regions):
        consumer = _consumer_surplus_proxy(dispatch, prices, flows, region, regions, period_weights)
        producer = _producer_surplus(capacities, dispatch, prices, region, regions, period_weights)
        periods = sorted(set(consumer) | set(producer) | set(congestion.get(region, {})))
        components[region] = {
            int(period): {
                "consumer_surplus": consumer.get(period, 0.0),
                "producer_surplus": producer.get(period, 0.0),
                "congestion_rent": congestion.get(region, {}).get(period, 0.0),
            }
            for period in periods
        }
    return components


def _find_scenario_dir(
    output_dir: Path,
    scenario_name: str,
    weather_year: int,
    climate_year_count: int,
) -> Path:
    return scenario_output_dir(output_dir, scenario_name, weather_year, climate_year_count)


def scenario_output_dir(
    output_root: Path,
    model_case: str,
    weather_year: int,
    climate_year_count: int,
) -> Path:
    if climate_year_count == 1:
        return output_root / model_case
    return output_root / model_case / f"weather_{weather_year}"


def load_welfare_effects(output_dir: Path = OUTPUT_DIR) -> list[dict]:
    period_weights = _load_period_weights()
    effects = []
    scenario_dirs = [
        path
        for model_dir in sorted(path for path in output_dir.iterdir() if path.is_dir())
        for path in (
            sorted(model_dir.iterdir())
            if any(child.is_dir() for child in model_dir.iterdir())
            else [model_dir]
        )
        if path.is_dir() and has_output_tables(path)
    ]
    for forced_dir in scenario_dirs:
        metadata_path = forced_dir / "model_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        forced_capacity = metadata.get("forced_capacity")
        base_model_case = metadata.get("base_model_case")
        weather_year = metadata.get("weather_year")
        if not forced_capacity or not base_model_case or weather_year is None:
            continue
        base_dir = Path(forced_capacity.get("base_output_dir", ""))
        if not base_dir.exists():
            base_dir = _find_scenario_dir(output_dir, base_model_case, int(weather_year), 3)
        if not has_output_tables(base_dir):
            continue

        base = _welfare_components(base_dir, period_weights)
        forced = _welfare_components(forced_dir, period_weights)
        rows = []
        for country in sorted(base):
            periods = sorted(set(base[country]) | set(forced.get(country, {})))
            for period in periods:
                delta = {
                    key: forced[country].get(period, {}).get(key, 0.0)
                    - base[country].get(period, {}).get(key, 0.0)
                    for key in ["consumer_surplus", "producer_surplus", "congestion_rent"]
                }
                delta["sew"] = sum(delta.values())
                rows.append(
                    {
                        "country": country,
                        "period": int(period),
                        "consumer_surplus_meur": _round(delta["consumer_surplus"] / 1_000_000, 3),
                        "producer_surplus_meur": _round(delta["producer_surplus"] / 1_000_000, 3),
                        "congestion_rent_meur": _round(delta["congestion_rent"] / 1_000_000, 3),
                        "sew_meur": _round(delta["sew"] / 1_000_000, 3),
                    }
                )
        for period in sorted({row["period"] for row in rows}):
            period_rows = [row for row in rows if row["period"] == period]
            rows.append(
                {
                    "country": "Total",
                    "period": int(period),
                    "consumer_surplus_meur": _round(
                        sum(row["consumer_surplus_meur"] for row in period_rows), 3
                    ),
                    "producer_surplus_meur": _round(
                        sum(row["producer_surplus_meur"] for row in period_rows), 3
                    ),
                    "congestion_rent_meur": _round(
                        sum(row["congestion_rent_meur"] for row in period_rows), 3
                    ),
                    "sew_meur": _round(sum(row["sew_meur"] for row in period_rows), 3),
                }
            )
        effects.append(
            {
                "scenario": metadata.get("active_model", forced_dir.parent.name),
                "baseScenario": base_model_case,
                "weatherYear": int(weather_year),
                "forcedCapacity": forced_capacity,
                "rows": rows,
            }
        )
    return effects


def _rows_by_timestep(rows: list[dict]) -> dict[str, dict]:
    return {str(row["timestep"]): row for row in rows}


def _aligned_week_window(
    reference_rows: list[dict],
    target_rows: list[dict],
    preset: str,
) -> dict:
    target_by_timestep = _rows_by_timestep(target_rows)
    aligned_rows = [target_by_timestep[row["timestep"]] for row in reference_rows if row["timestep"] in target_by_timestep]
    label = ""
    if aligned_rows:
        start = str(aligned_rows[0]["timestep"])[:10]
        end = str(aligned_rows[-1]["timestep"])[:10]
        label = f"{start} to {end}"
    return {"label": label, "preset": preset, "rows": aligned_rows}


def _outage_week_data(with_link: dict, islanded: dict, period: str) -> dict:
    with_link_rows = with_link["weekData"][period]
    islanded_rows = islanded["weekData"][period]
    islanded_frame = pd.DataFrame(islanded_rows)
    if islanded_frame.empty:
        return {"withInterconnector": {}, "islanded": {}}
    islanded_frame["timestep"] = pd.to_datetime(islanded_frame["timestep"])
    windows = {
        "winter": _fixed_week(islanded_frame, "01-15"),
        "summer": _fixed_week(islanded_frame, "07-15"),
        "shed": _highest_shedding_week(islanded_frame),
    }
    islanded_windows = {
        key: {
            "label": _week_label(window),
            "preset": key,
            "rows": _series_points(window, WEEK_COLUMNS),
        }
        for key, window in windows.items()
    }
    with_link_windows = {
        key: _aligned_week_window(window["rows"], with_link_rows, key)
        for key, window in islanded_windows.items()
    }
    return {"withInterconnector": with_link_windows, "islanded": islanded_windows}


def load_outage_comparison(output_dir: Path = OUTPUT_DIR) -> dict | None:
    period = "2030"
    years = {}
    for with_link_dir in sorted((output_dir / "DK_NL").glob("weather_*")):
        if not has_output_tables(with_link_dir):
            continue
        weather_year = int(with_link_dir.name.removeprefix("weather_"))
        islanded_name = f"DK_dispatch_fixed_2030_from_DK_NL_DK_weather_{weather_year}"
        islanded_dir = scenario_output_dir(output_dir, islanded_name, weather_year, 3)
        if not has_output_tables(islanded_dir):
            continue
        with_link = load_dashboard_data(with_link_dir, compact_week_data=False)["regions"]["DK"]
        islanded = load_dashboard_data(islanded_dir, compact_week_data=False)
        years[str(weather_year)] = {
            "period": period,
            "weatherYear": weather_year,
            "withInterconnectorLabel": "DK_NL with interconnector",
            "islandedLabel": "DK islanded dispatch",
            "islandedScenario": islanded_name,
            "weekData": _outage_week_data(with_link, islanded, period),
        }
    if not years:
        return None
    sorted_years = sorted(years, key=int)
    return {
        "years": years,
        "defaultClimateYear": "2008" if "2008" in years else sorted_years[0],
    }


def load_dashboard_bundle(output_dir: Path = OUTPUT_DIR, compact_week_data: bool = True) -> dict:
    datasets = {}

    def add_dataset(scenario_dir: Path) -> None:
        data = load_dashboard_data(scenario_dir, compact_week_data=compact_week_data)
        country = str(data["metadata"].get("active_model") or scenario_dir.name)
        climate_year = str(data["metadata"].get("weather_year") or scenario_dir.name)
        datasets.setdefault(country, {})[climate_year] = data

    for country_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        weather_dirs = [
            path
            for path in sorted(country_dir.iterdir())
            if path.is_dir() and has_output_tables(path)
        ]
        if weather_dirs:
            for weather_dir in weather_dirs:
                add_dataset(weather_dir)
        elif has_output_tables(country_dir):
            add_dataset(country_dir)

    if not datasets and has_output_tables(output_dir):
        add_dataset(output_dir)

    if not datasets:
        raise FileNotFoundError(f"No model output tables found under {output_dir}")

    preferred = "DK_NL" if "DK_NL" in datasets else sorted(datasets)[0]
    preferred_years = sorted(datasets[preferred], key=int)
    preferred_climate = "2008" if "2008" in preferred_years else preferred_years[0]
    return {
        "datasets": datasets,
        "defaultCountry": preferred,
        "defaultClimateYear": preferred_climate,
        "welfareEffects": load_welfare_effects(output_dir),
        "outageComparison": load_outage_comparison(output_dir),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pathway Pilot - PyPSA Outputs</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1d252d;
      --muted: #65707d;
      --line: #d9e0e6;
      --wind: #2f80ed;
      --solar: #f2b705;
      --gas: #7a5195;
      --gas-cc: #00a6a6;
      --shed: #c43d3d;
      --demand: #111827;
      --residual: #18a058;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
      line-height: 1.45;
    }
    header {
      padding: 30px 34px 24px;
      background: #17202a;
      color: white;
      border-bottom: 4px solid #38a3a5;
    }
    header h1 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }
    header p {
      margin: 8px 0 0;
      color: #cbd6df;
      max-width: 1000px;
    }
    main {
      max-width: 1380px;
      margin: 0 auto;
      padding: 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 22px rgba(28, 35, 45, 0.04);
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      align-items: start;
    }
    .panel {
      padding: 16px;
      min-width: 0;
    }
    .panel.wide { grid-column: 1 / -1; }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 17px;
      letter-spacing: 0;
    }
    .security-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    .security-table th, .security-table td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: right;
    }
    .security-table th:first-child, .security-table td:first-child {
      text-align: left;
    }
    .metric-tip {
      cursor: help;
      border-bottom: 1px dotted var(--muted);
    }
    .controls {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 12px;
    }
    .tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 16px;
    }
    .tab-button[aria-selected="true"] {
      border-color: #17202a;
      background: #17202a;
      color: white;
    }
    .tab-panel[hidden] {
      display: none;
    }
    .region-picker {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .region-picker label {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      padding: 6px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--ink);
    }
    .region-picker input {
      margin: 0;
      width: 14px;
      height: 14px;
      padding: 0;
      accent-color: #17202a;
    }
    label {
      color: var(--muted);
      font-size: 13px;
    }
    input, select, button {
      border: 1px solid var(--line);
      background: white;
      border-radius: 6px;
      color: var(--ink);
      padding: 7px 9px;
      font: inherit;
    }
    button {
      cursor: pointer;
      color: var(--ink);
    }
    button:hover {
      border-color: #9fb0bf;
      background: #f8fafb;
    }
    button.active {
      border-color: #17202a;
      background: #17202a;
      color: white;
    }
    svg {
      display: block;
      width: 100%;
      height: 330px;
    }
    .legend {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }
    .muted-inline {
      color: var(--muted);
      font-size: 13px;
    }
    .swatch {
      width: 11px;
      height: 11px;
      display: inline-block;
      border-radius: 2px;
      margin-right: 5px;
      vertical-align: -1px;
    }
    .tooltip {
      position: fixed;
      pointer-events: none;
      opacity: 0;
      max-width: 280px;
      padding: 9px 10px;
      color: white;
      background: rgba(23, 32, 42, 0.96);
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.4;
      z-index: 10;
      transform: translate(12px, 12px);
      transition: opacity 0.08s ease;
    }
    .axis text, .tick { fill: #67717d; font-size: 11px; }
    .axis line, .axis path, .gridline { stroke: #dfe5ea; }
    .empty {
      color: var(--muted);
      padding: 48px 0;
      text-align: center;
    }
    @media (max-width: 920px) {
      main { padding: 14px; }
      header { padding: 24px 18px; }
      .grid { grid-template-columns: 1fr; }
      svg { height: 290px; }
    }
  </style>
</head>
<body>
<header>
  <h1>Pathway Pilot - PyPSA Outputs</h1>
  <p id="modelSummary">One-zone capacity expansion and dispatch results. The 2050 time series reuses 2040 profiles.</p>
</header>
<main>
  <div class="tabs" role="tablist" aria-label="Dashboard views">
    <button class="tab-button" id="detailsTab" type="button" role="tab" aria-selected="true" aria-controls="detailsPanel">Scenario Details</button>
    <button class="tab-button" id="compareTab" type="button" role="tab" aria-selected="false" aria-controls="comparePanel">DK/NL Comparison</button>
    <button class="tab-button" id="outageTab" type="button" role="tab" aria-selected="false" aria-controls="outagePanel">DK Interconnector Outage</button>
    <button class="tab-button" id="welfareTab" type="button" role="tab" aria-selected="false" aria-controls="welfarePanel">SEW Effects</button>
  </div>
  <section id="detailsPanel" class="tab-panel" role="tabpanel" aria-labelledby="detailsTab">
    <div class="controls">
      <label for="countrySelect">Scenario</label>
      <select id="countrySelect"></select>
      <label for="climateYearSelect">Climate year</label>
      <select id="climateYearSelect"></select>
      <label id="regionSelectLabel" for="regionSelect" hidden>Region</label>
      <select id="regionSelect" hidden></select>
    </div>
    <div class="grid">
    <div class="panel">
      <h2>Active capacity by model year</h2>
      <svg id="capacityChart"></svg>
      <div class="legend" id="capacityLegend"></div>
    </div>
    <div class="panel">
      <h2>Capacity built by investment year</h2>
      <svg id="buildCapacityChart"></svg>
      <div class="legend" id="buildCapacityLegend"></div>
    </div>
    <div class="panel">
      <h2>Average prices</h2>
      <svg id="averagePriceChart"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:var(--demand)"></span>Average electricity price</span>
      </div>
    </div>
    <div class="panel">
      <h2>Duration curve</h2>
      <svg id="durationChart"></svg>
      <div class="legend" id="durationLegend"></div>
    </div>
    <div class="panel wide">
      <h2>Dispatch week</h2>
      <div class="controls">
        <label for="weekPeriod">Model year</label>
        <select id="weekPeriod"></select>
        <label id="weekStartLabel" for="weekStart">Week start</label>
        <input id="weekStart" type="date">
        <button id="winterWeek" type="button">Winter week</button>
        <button id="summerWeek" type="button">Summer week</button>
        <button id="shedWeek" type="button">Highest shedding</button>
        <span id="weekRangeLabel" class="muted-inline"></span>
      </div>
      <svg id="weekChart"></svg>
      <div class="legend" id="weekLegend">
        <span><span class="swatch" style="background:var(--gas-cc)"></span>Gas turbine CC</span>
        <span><span class="swatch" style="background:var(--gas)"></span>Gas turbine</span>
        <span><span class="swatch" style="background:var(--wind)"></span>Wind</span>
        <span><span class="swatch" style="background:var(--solar)"></span>Solar</span>
        <span><span class="swatch" style="background:var(--shed)"></span>Load shedding</span>
        <span><span class="swatch" style="background:var(--demand)"></span>Demand</span>
      </div>
    </div>
    <div class="panel wide">
      <h2>Security of supply</h2>
      <div id="securityTable"></div>
    </div>
    <div class="panel">
      <h2>Capture rates</h2>
      <svg id="captureChart"></svg>
      <div class="legend" id="captureLegend"></div>
    </div>
    <div class="panel">
      <h2>Unit CAPEX by investment year</h2>
      <svg id="capexChart"></svg>
      <div class="legend" id="capexLegend"></div>
    </div>
    </div>
  </section>
  <section id="comparePanel" class="tab-panel" role="tabpanel" aria-labelledby="compareTab" hidden>
    <div class="controls">
      <label>Regions</label>
      <div class="region-picker" id="comparisonRegionPicker"></div>
      <label for="comparisonClimateYearSelect">Climate year</label>
      <select id="comparisonClimateYearSelect"></select>
    </div>
    <div class="grid">
      <div class="panel">
        <h2>Annual baseload prices</h2>
        <svg id="comparisonPriceChart"></svg>
        <div class="legend" id="comparisonPriceLegend"></div>
      </div>
      <div class="panel">
        <h2>Generation shares</h2>
        <svg id="comparisonGenerationChart"></svg>
        <div class="legend" id="comparisonGenerationLegend"></div>
      </div>
    </div>
  </section>
  <section id="welfarePanel" class="tab-panel" role="tabpanel" aria-labelledby="welfareTab" hidden>
    <div class="controls">
      <label for="welfareScenarioSelect">Forced scenario</label>
      <select id="welfareScenarioSelect"></select>
    </div>
    <div class="grid">
      <div class="panel wide">
        <h2>Socio-economic welfare decomposition</h2>
        <div id="welfareSummary"></div>
      </div>
      <div class="panel wide">
        <h2>SEW contributions by country</h2>
        <svg id="welfareChart"></svg>
        <div class="legend" id="welfareLegend"></div>
      </div>
    </div>
  </section>
  <section id="outagePanel" class="tab-panel" role="tabpanel" aria-labelledby="outageTab" hidden>
    <div class="grid">
      <div class="panel wide">
        <h2>DK 2030 security of supply with and without NL interconnector</h2>
        <div class="controls">
          <label for="outageClimateYearSelect">Climate year</label>
          <select id="outageClimateYearSelect"></select>
        </div>
        <div id="outageComparisonTable"></div>
      </div>
      <div class="panel wide">
        <h2>Dispatch week - DK_NL with interconnector</h2>
        <div class="controls">
          <button id="outageWinterWeek" type="button">Winter week</button>
          <button id="outageSummerWeek" type="button">Summer week</button>
          <button id="outageShedWeek" type="button">Highest shedding</button>
          <span id="outageWeekRangeLabel" class="muted-inline"></span>
        </div>
        <svg id="outageWithLinkWeekChart"></svg>
        <div class="legend" id="outageWithLinkWeekLegend"></div>
      </div>
      <div class="panel wide">
        <h2>Dispatch week - DK islanded dispatch</h2>
        <svg id="outageIslandedWeekChart"></svg>
        <div class="legend" id="outageIslandedWeekLegend"></div>
      </div>
    </div>
  </section>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const APP_DATA = __DATA__;
const DATASETS = APP_DATA.datasets || {};
const WELFARE_EFFECTS = APP_DATA.welfareEffects || [];
const OUTAGE_COMPARISON = APP_DATA.outageComparison || null;
let DATA;
let activeWeekPreset = "shed";
let activeOutageWeekPreset = "shed";
const COLORS = { wind: "#2f80ed", solar: "#f2b705", gas: "#7a5195", gas_turbine_cc: "#00a6a6", load_shedding: "#c43d3d", demand: "#111827", residual_load: "#18a058", interconnector_import: "#8a8f98", interconnector_export: "#ef8354", consumer_surplus: "#2f80ed", producer_surplus: "#18a058", congestion_rent: "#ef8354", sew: "#111827" };
const YEAR_COLORS = ["#111827", "#2f80ed", "#18a058", "#c43d3d", "#7a5195"];
const COUNTRY_COLORS = ["#111827", "#2f80ed", "#18a058", "#c43d3d", "#7a5195", "#00a6a6"];
const LABELS = { wind: "Wind", solar: "Solar", gas: "Gas turbine", gas_turbine_cc: "Gas turbine CC", load_shedding: "Load shedding", demand: "Demand", interconnector_import: "Import", interconnector_export: "Export" };
const tooltip = document.getElementById("tooltip");
const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const fmt0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const fmt1 = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const fmt2 = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
function labelFor(key) { return LABELS[key] || key.replaceAll("_", " "); }
function generationShareLabel(key) { return key === "gas" ? "Gas" : labelFor(key); }
function renderModelSummary() {
  const meta = DATA.metadata || {};
  const demand = (meta.demand_zones || ["DKE1", "DKW1"]).join(" + ");
  const cfZone = meta.capacity_factor_zone || "DKW1";
  const model = meta.active_model || "DK";
  const region = meta.selected_region ? `, region ${meta.selected_region}` : "";
  if (meta.model_regions && meta.selected_region) {
    const regionMeta = meta.model_regions[meta.selected_region] || {};
    const regionDemand = (regionMeta.demand_zones || []).join(" + ");
    document.getElementById("modelSummary").textContent =
      `${model}${region} capacity expansion and dispatch results. Capacity factors use ${regionMeta.capacity_factor_zone}; demand is ${regionDemand}. The 2050 time series reuses 2040 profiles.`;
    return;
  }
  document.getElementById("modelSummary").textContent =
    `One-zone ${model} capacity expansion and dispatch results. Capacity factors use ${cfZone}; demand is ${demand}. The 2050 time series reuses 2040 profiles.`;
}

function initCountryControl() {
  const select = document.getElementById("countrySelect");
  const countries = Object.keys(DATASETS).sort();
  select.innerHTML = countries.map(country => `<option value="${country}">${country}</option>`).join("");
  select.value = APP_DATA.defaultCountry && DATASETS[APP_DATA.defaultCountry] ? APP_DATA.defaultCountry : countries[0];
  refreshClimateControl(APP_DATA.defaultClimateYear);
  setCurrentDataset();
  select.addEventListener("change", () => {
    refreshClimateControl();
    setCurrentDataset();
    rerenderForScenario();
  });
  document.getElementById("climateYearSelect").addEventListener("change", () => {
    refreshRegionControl();
    setCurrentDataset();
    rerenderForScenario();
  });
  document.getElementById("regionSelect").addEventListener("change", () => {
    setCurrentDataset();
    rerenderForScenario();
  });
}

function refreshClimateControl(preferredYear) {
  const country = document.getElementById("countrySelect").value;
  const select = document.getElementById("climateYearSelect");
  const years = Object.keys(DATASETS[country] || {}).sort((a, b) => Number(a) - Number(b));
  select.innerHTML = years.map(year => `<option value="${year}">${year}</option>`).join("");
  select.value = preferredYear && years.includes(String(preferredYear))
    ? String(preferredYear)
    : (years.includes("2008") ? "2008" : years[0]);
  refreshRegionControl();
}

function currentScenarioDataset() {
  const country = document.getElementById("countrySelect").value;
  const climateYear = document.getElementById("climateYearSelect").value;
  return DATASETS[country]?.[climateYear];
}

function refreshRegionControl() {
  const scenario = currentScenarioDataset();
  const select = document.getElementById("regionSelect");
  const label = document.getElementById("regionSelectLabel");
  const regions = scenario?.regions ? Object.keys(scenario.regions) : [];
  const show = regions.length > 0;
  select.hidden = !show;
  label.hidden = !show;
  if (!show) {
    select.innerHTML = "";
    return;
  }
  const previous = select.value;
  select.innerHTML = regions.map(region => `<option value="${region}">${region}</option>`).join("");
  select.value = regions.includes(previous)
    ? previous
    : (scenario.defaultRegion && regions.includes(scenario.defaultRegion) ? scenario.defaultRegion : regions[0]);
}

function setCurrentDataset() {
  const country = document.getElementById("countrySelect").value;
  const climateYear = document.getElementById("climateYearSelect").value;
  const scenario = DATASETS[country][climateYear];
  const region = document.getElementById("regionSelect").value;
  DATA = scenario.regions && scenario.regions[region] ? scenario.regions[region] : scenario;
}

function rerenderForScenario() {
  initControls();
  renderModelSummary();
  renderAll();
}

function selectedComparisonRegions() {
  return [...document.querySelectorAll("#comparisonRegionPicker input:checked")]
    .map(input => input.value)
    .filter(country => DATASETS[country]);
}

function comparisonClimateYear() {
  return document.getElementById("comparisonClimateYearSelect").value;
}

function initComparisonControls() {
  const countries = Object.keys(DATASETS).sort();
  const picker = document.getElementById("comparisonRegionPicker");
  const defaultCountries = countries.filter(country => ["DK", "NL"].includes(country));
  const selected = defaultCountries.length ? defaultCountries : countries;
  picker.innerHTML = countries.map(country => `
    <label>
      <input type="checkbox" value="${country}" ${selected.includes(country) ? "checked" : ""}>
      ${country}
    </label>
  `).join("");
  picker.querySelectorAll("input").forEach(input => input.addEventListener("change", renderComparison));

  const climateSelect = document.getElementById("comparisonClimateYearSelect");
  const climateYears = [...new Set(countries.flatMap(country => Object.keys(DATASETS[country] || {})))]
    .sort((a, b) => Number(a) - Number(b));
  climateSelect.innerHTML = climateYears.map(year => `<option value="${year}">${year}</option>`).join("");
  climateSelect.value = APP_DATA.defaultClimateYear && climateYears.includes(String(APP_DATA.defaultClimateYear))
    ? String(APP_DATA.defaultClimateYear)
    : (climateYears.includes("2008") ? "2008" : climateYears[0]);
  climateSelect.addEventListener("change", renderComparison);
}

function initWelfareControls() {
  const select = document.getElementById("welfareScenarioSelect");
  if (!WELFARE_EFFECTS.length) {
    select.innerHTML = "";
    return;
  }
  select.innerHTML = WELFARE_EFFECTS.map((effect, index) =>
    `<option value="${index}">${effect.scenario} vs ${effect.baseScenario}, ${effect.weatherYear}</option>`
  ).join("");
  select.value = "0";
  select.addEventListener("change", renderWelfare);
}

function showTip(event, html) {
  tooltip.innerHTML = html;
  tooltip.style.left = event.clientX + "px";
  tooltip.style.top = event.clientY + "px";
  tooltip.style.opacity = 1;
}
function hideTip() { tooltip.style.opacity = 0; }
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
function addEmpty(svg, message) {
  const text = svgEl("text", { x: 70, y: 90, class: "tick" });
  text.textContent = message;
  svg.appendChild(text);
}
function dims(svg) {
  const box = svg.getBoundingClientRect();
  return { width: Math.max(box.width, 320), height: Math.max(box.height, 260), margin: { top: 18, right: 20, bottom: 38, left: 58 } };
}
function scale(domainMin, domainMax, rangeMin, rangeMax) {
  const span = domainMax - domainMin || 1;
  return value => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}
function niceUpperBound(value) {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const exponent = Math.floor(Math.log10(value));
  const base = Math.pow(10, exponent);
  const normalized = value / base;
  const niceNormalized = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10;
  return niceNormalized * base;
}
function addYAxis(svg, x, yScale, max, plotTop, plotBottom, formatTick, label) {
  const width = svg.getBoundingClientRect().width;
  for (let i = 0; i <= 4; i++) {
    const value = max * i / 4;
    const y = yScale(value);
    svg.appendChild(svgEl("line", { x1: x, x2: width - 20, y1: y, y2: y, class: "gridline" }));
    const text = svgEl("text", { x: x - 8, y: y + 4, "text-anchor": "end", class: "tick" });
    text.textContent = formatTick(value);
    svg.appendChild(text);
  }
  if (label) {
    const unit = svgEl("text", { x: x, y: 12, "text-anchor": "start", class: "tick" });
    unit.textContent = label;
    svg.appendChild(unit);
  }
}
function pathLine(points, x, y) {
  return points.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`).join(" ");
}

function initTabs() {
  const tabs = [
    { button: document.getElementById("detailsTab"), panel: document.getElementById("detailsPanel") },
    { button: document.getElementById("compareTab"), panel: document.getElementById("comparePanel") },
    { button: document.getElementById("outageTab"), panel: document.getElementById("outagePanel") },
    { button: document.getElementById("welfareTab"), panel: document.getElementById("welfarePanel") }
  ];
  tabs.forEach(tab => {
    tab.button.addEventListener("click", () => {
      tabs.forEach(item => {
        const selected = item === tab;
        item.button.setAttribute("aria-selected", selected ? "true" : "false");
        item.panel.hidden = !selected;
      });
      if (tab.panel.id === "comparePanel") renderComparison();
      if (tab.panel.id === "outagePanel") renderOutageComparison();
      if (tab.panel.id === "welfarePanel") renderWelfare();
    });
  });
}

function renderSecurity() {
  const definitions = {
    lole: "LOLE (Loss of Load Expectation): number of hours in the simulated year with non-zero load shedding. This deterministic model-year value is reported as an LOLE-style metric.",
    eens: "EENS (Expected Energy Not Served): total unserved electricity demand in the simulated year. This deterministic model-year value is reported as an EENS-style metric.",
    eensPct: "EENS as percent of annual demand: total unserved electricity demand divided by total annual electricity demand in the simulated model year.",
    peak: "Peak shed: maximum hourly load shedding in the simulated year.",
    gas: "Total gas turbine generation: annual electricity output from simple-cycle and combined-cycle gas turbines in the simulated model year.",
    imports: "Annual electricity imported through modelled interconnectors. This is non-zero only for regional views of combined models with interconnector flow output.",
    exports: "Annual electricity exported through modelled interconnectors. This is non-zero only for regional views of combined models with interconnector flow output.",
    interconnectorUtilisation: "Interconnector utilisation: annual absolute flow, imports plus exports, divided by interconnector capacity times simulated hours.",
    voll: "VoLL (Value of Lost Load): penalty cost assigned to involuntary load shedding in the optimisation, expressed in EUR/MWh."
  };
  const rows = DATA.security.map(row => `
    <tr>
      <td>${row.period}</td>
      <td>${fmt1.format((DATA.summary.find(item => item.period === row.period)?.peak_demand_mw || 0) / 1000)} GW</td>
      <td>${fmt.format(DATA.summary.find(item => item.period === row.period)?.energy_twh || 0)} TWh</td>
      <td><span class="metric-tip" data-tip="${definitions.lole}">${fmt0.format(row.lole_hours)}</span></td>
      <td><span class="metric-tip" data-tip="${definitions.eens}">${fmt2.format(row.eens_gwh)}</span></td>
      <td><span class="metric-tip" data-tip="${definitions.eensPct}">${fmt2.format(row.eens_pct_demand)}</span></td>
      <td><span class="metric-tip" data-tip="${definitions.peak}">${fmt1.format(row.peak_shed_mw / 1000)}</span></td>
      <td><span class="metric-tip" data-tip="${definitions.gas}">${fmt1.format(row.gas_generation_twh)}</span></td>
      <td><span class="metric-tip" data-tip="${definitions.imports}">${fmt1.format(row.import_twh || 0)}</span></td>
      <td><span class="metric-tip" data-tip="${definitions.exports}">${fmt1.format(row.export_twh || 0)}</span></td>
      <td><span class="metric-tip" data-tip="${definitions.interconnectorUtilisation}">${fmt1.format(row.interconnector_utilisation_pct || 0)}</span></td>
      <td><span class="metric-tip" data-tip="${definitions.voll}">${fmt0.format(row.voll_eur_per_mwh)}</span></td>
    </tr>
  `).join("");
  document.getElementById("securityTable").innerHTML = `
    <table class="security-table">
      <thead>
        <tr>
          <th>Model year</th>
          <th>Peak demand</th>
          <th>Annual demand</th>
          <th><span class="metric-tip" data-tip="${definitions.lole}">LOLE</span></th>
          <th><span class="metric-tip" data-tip="${definitions.eens}">EENS</span></th>
          <th><span class="metric-tip" data-tip="${definitions.eensPct}">EENS / demand</span></th>
          <th><span class="metric-tip" data-tip="${definitions.peak}">Peak shed</span></th>
          <th><span class="metric-tip" data-tip="${definitions.gas}">Gas generation</span></th>
          <th><span class="metric-tip" data-tip="${definitions.imports}">Import</span></th>
          <th><span class="metric-tip" data-tip="${definitions.exports}">Export</span></th>
          <th><span class="metric-tip" data-tip="${definitions.interconnectorUtilisation}">Interconnector utilisation</span></th>
          <th><span class="metric-tip" data-tip="${definitions.voll}">VoLL</span></th>
        </tr>
        <tr>
          <th></th>
          <th>GW</th>
          <th>TWh/y</th>
          <th>h/y</th>
          <th>GWh/y</th>
          <th>%</th>
          <th>GW</th>
          <th>TWh/y</th>
          <th>TWh/y</th>
          <th>TWh/y</th>
          <th>%</th>
          <th>EUR/MWh</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
  document.querySelectorAll(".metric-tip").forEach(el => {
    el.addEventListener("mousemove", e => showTip(e, el.dataset.tip));
    el.addEventListener("mouseleave", hideTip);
  });
}

function securityRowFor(dataset, period) {
  const security = (dataset?.security || []).find(row => Number(row.period) === Number(period));
  if (!security) return null;
  const summary = (dataset.summary || []).find(row => Number(row.period) === Number(period)) || {};
  return { ...security, peak_demand_mw: summary.peak_demand_mw || 0, energy_twh: summary.energy_twh || 0 };
}

function outageClimateYear() {
  return document.getElementById("outageClimateYearSelect").value;
}

function selectedOutageComparison() {
  return OUTAGE_COMPARISON?.years?.[outageClimateYear()] || null;
}

function renderOutageComparison() {
  const container = document.getElementById("outageComparisonTable");
  const comparison = selectedOutageComparison();
  const year = outageClimateYear();
  const interconnectorDataset = DATASETS.DK_NL?.[year]?.regions?.DK;
  const islandedDataset = DATASETS[comparison?.islandedScenario]?.[year];
  const interconnector = securityRowFor(interconnectorDataset, 2030);
  const islanded = securityRowFor(islandedDataset, 2030);
  if (!interconnector || !islanded) {
    container.innerHTML = `
      <div class="empty">
        Missing DK_NL 2008 DK-region or islanded fixed-capacity dispatch output for 2030.
      </div>`;
    return;
  }
  const metrics = [
    { label: "Peak demand", unit: "GW", value: row => row.peak_demand_mw / 1000, formatter: fmt1 },
    { label: "Annual demand", unit: "TWh/y", value: row => row.energy_twh, formatter: fmt1 },
    { label: "LOLE", unit: "h/y", value: row => row.lole_hours, formatter: fmt0 },
    { label: "EENS", unit: "GWh/y", value: row => row.eens_gwh, formatter: fmt2 },
    { label: "EENS / demand", unit: "%", value: row => row.eens_pct_demand, formatter: fmt2 },
    { label: "Peak shed", unit: "GW", value: row => row.peak_shed_mw / 1000, formatter: fmt1 },
    { label: "Gas generation", unit: "TWh/y", value: row => row.gas_generation_twh, formatter: fmt1 },
    { label: "Import", unit: "TWh/y", value: row => row.import_twh || 0, formatter: fmt1 },
    { label: "Export", unit: "TWh/y", value: row => row.export_twh || 0, formatter: fmt1 },
    { label: "Interconnector utilisation", unit: "%", value: row => row.interconnector_utilisation_pct || 0, formatter: fmt1 },
    { label: "Average price across all hours", unit: "EUR/MWh", value: row => row.average_price_eur_per_mwh || 0, formatter: fmt1 },
    { label: "Average price without load shedding", unit: "EUR/MWh", value: row => row.average_price_without_shedding_eur_per_mwh || 0, formatter: fmt1 },
    { label: "VoLL", unit: "EUR/MWh", value: row => row.voll_eur_per_mwh, formatter: fmt0 },
    {
      label: "Cumulative VoLL",
      unit: "MEUR/y",
      value: row => (row.eens_mwh || 0) * (row.voll_eur_per_mwh || 0) / 1_000_000,
      formatter: fmt1
    }
  ];
  const rows = metrics.map(metric => {
    const withLink = metric.value(interconnector);
    const island = metric.value(islanded);
    const delta = island - withLink;
    return `
      <tr>
        <td>${metric.label}</td>
        <td>${metric.unit}</td>
        <td>${metric.formatter.format(withLink)}</td>
        <td>${metric.formatter.format(island)}</td>
        <td>${metric.formatter.format(delta)}</td>
      </tr>`;
  }).join("");
  container.innerHTML = `
    <p class="muted-inline">
      Compares DK in 2030 under the combined DK_NL ${year} model with the fixed-capacity DK-only dispatch run using the DK_NL ${year} DK build-out.
    </p>
    <table class="security-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Unit</th>
          <th>DK_NL with interconnector</th>
          <th>DK islanded dispatch</th>
          <th>Islanded minus with interconnector</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
  renderOutageWeekCharts();
}

function outageDispatchCarriers(rows) {
  const preferred = ["gas_turbine_cc", "gas", "wind", "solar", "interconnector_import", "interconnector_export", "load_shedding"];
  return preferred.filter(carrier => rows.some(row => Math.abs(row[carrier] || 0) > 1e-6 || carrier === "load_shedding"));
}

function renderDispatchWeekChart(svgId, legendId, rows, carriers, period) {
  const svg = document.getElementById(svgId); clear(svg);
  const legend = document.getElementById(legendId);
  if (!rows.length) {
    addEmpty(svg, "No data for selected week.");
    legend.innerHTML = "";
    return;
  }
  const { width, height, margin } = dims(svg);
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const positiveTotals = rows.map(r => carriers.reduce((sum, carrier) => sum + Math.max(r[carrier] || 0, 0), 0) / 1000);
  const negativeTotals = rows.map(r => carriers.reduce((sum, carrier) => sum + Math.min(r[carrier] || 0, 0), 0) / 1000);
  const maxY = Math.max(...rows.map(r => r.demand / 1000), ...positiveTotals, 1) * 1.05;
  const minY = Math.min(...negativeTotals, 0) * 1.05;
  const x = scale(0, rows.length - 1, margin.left, margin.left + plotW);
  const y = scale(minY, maxY, margin.top + plotH, margin.top);
  const axisMax = Math.max(Math.abs(minY), Math.abs(maxY));
  addYAxis(svg, margin.left, y, maxY, margin.top, margin.top + plotH, v => fmt1.format(v), "GW");
  if (minY < 0) {
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y(0), y2: y(0), stroke: "#7c8794", "stroke-width": 1.2 }));
    for (let i = 1; i <= 2; i++) {
      const value = -axisMax * i / 2;
      const text = svgEl("text", { x: margin.left - 8, y: y(value) + 4, "text-anchor": "end", class: "tick" });
      text.textContent = fmt1.format(value);
      svg.appendChild(text);
    }
  }
  const band = plotW / rows.length;
  rows.forEach((r, i) => {
    let positiveY0 = 0;
    let negativeY0 = 0;
    carriers.forEach(carrier => {
      const value = (r[carrier] || 0) / 1000;
      const base = value >= 0 ? positiveY0 : negativeY0;
      const next = base + value;
      const rect = svgEl("rect", {
        x: x(i) - band * 0.45,
        y: value >= 0 ? y(next) : y(base),
        width: Math.max(band * 0.9, 1),
        height: Math.max(Math.abs(y(base) - y(next)), 0),
        fill: COLORS[carrier]
      });
      const dispatchLines = carriers.map(name => `${labelFor(name)}: ${fmt1.format((r[name] || 0) / 1000)} GW`).join("<br>");
      rect.addEventListener("mousemove", e => showTip(e, `<b>${period} ${r.timestep}</b><br>Demand: ${fmt1.format(r.demand / 1000)} GW<br>${dispatchLines}<br>Price: ${fmt2.format(r.price_eur_per_mwh)} EUR/MWh`));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
      if (value >= 0) positiveY0 = next;
      else negativeY0 = next;
    });
  });
  const demandLine = rows.map((r, i) => ({ x: i, y: r.demand / 1000 }));
  svg.appendChild(svgEl("path", { d: pathLine(demandLine, x, y), fill: "none", stroke: COLORS.demand, "stroke-width": 2.2 }));
  legend.innerHTML = [...carriers, "demand"]
    .map(carrier => `<span><span class="swatch" style="background:${COLORS[carrier]}"></span>${labelFor(carrier)}</span>`)
    .join("");
}

function setActiveOutageWeekPreset(preset) {
  activeOutageWeekPreset = preset;
  const buttons = {
    winter: document.getElementById("outageWinterWeek"),
    summer: document.getElementById("outageSummerWeek"),
    shed: document.getElementById("outageShedWeek")
  };
  Object.entries(buttons).forEach(([key, button]) => {
    const isActive = key === preset;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

function renderOutageWeekCharts() {
  const comparison = selectedOutageComparison();
  const label = document.getElementById("outageWeekRangeLabel");
  if (!comparison?.weekData) {
    addEmpty(document.getElementById("outageWithLinkWeekChart"), "No outage comparison data is available.");
    addEmpty(document.getElementById("outageIslandedWeekChart"), "No outage comparison data is available.");
    label.textContent = "";
    return;
  }
  const preset = activeOutageWeekPreset || "shed";
  const withLink = comparison.weekData.withInterconnector?.[preset] || { rows: [], label: "" };
  const islanded = comparison.weekData.islanded?.[preset] || { rows: [], label: "" };
  label.textContent = islanded.label ? `Showing ${islanded.label}` : "";
  renderDispatchWeekChart(
    "outageWithLinkWeekChart",
    "outageWithLinkWeekLegend",
    withLink.rows || [],
    outageDispatchCarriers(withLink.rows || []),
    comparison.period || "2030"
  );
  renderDispatchWeekChart(
    "outageIslandedWeekChart",
    "outageIslandedWeekLegend",
    islanded.rows || [],
    outageDispatchCarriers(islanded.rows || []),
    comparison.period || "2030"
  );
}

function initOutageControls() {
  const select = document.getElementById("outageClimateYearSelect");
  const years = Object.keys(OUTAGE_COMPARISON?.years || {}).sort((a, b) => Number(a) - Number(b));
  select.innerHTML = years.map(year => `<option value="${year}">${year}</option>`).join("");
  select.value = OUTAGE_COMPARISON?.defaultClimateYear && years.includes(String(OUTAGE_COMPARISON.defaultClimateYear))
    ? String(OUTAGE_COMPARISON.defaultClimateYear)
    : (years.includes("2008") ? "2008" : years[0]);
  select.addEventListener("change", renderOutageComparison);
  setActiveOutageWeekPreset(activeOutageWeekPreset);
  document.getElementById("outageWinterWeek").onclick = () => {
    setActiveOutageWeekPreset("winter");
    renderOutageWeekCharts();
  };
  document.getElementById("outageSummerWeek").onclick = () => {
    setActiveOutageWeekPreset("summer");
    renderOutageWeekCharts();
  };
  document.getElementById("outageShedWeek").onclick = () => {
    setActiveOutageWeekPreset("shed");
    renderOutageWeekCharts();
  };
}

function renderStackedBars(svgId, rows, xKey, yKey, groups, labelKey, legendId, options = {}) {
  const svg = document.getElementById(svgId); clear(svg);
  const { width, height, margin } = dims(svg);
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const labels = [...new Set(rows.map(r => r[xKey]))].sort();
  const divisor = options.divisor || 1;
  const totals = labels.map(label => groups.reduce((sum, group) => sum + ((rows.find(r => r[xKey] === label && r[labelKey] === group)?.[yKey] || 0) / divisor), 0));
  const max = Math.max(...totals, 1) * 1.08;
  const y = scale(0, max, margin.top + plotH, margin.top);
  addYAxis(svg, margin.left, y, max, margin.top, margin.top + plotH, options.formatTick || (v => fmt0.format(v)), options.axisLabel || "");
  const band = plotW / labels.length;
  labels.forEach((label, i) => {
    let y0 = 0;
    groups.forEach(group => {
      const row = rows.find(r => r[xKey] === label && r[labelKey] === group);
      const rawValue = row ? row[yKey] : 0;
      const value = rawValue / divisor;
      const rectY = y(y0 + value), rectH = y(y0) - y(y0 + value);
      const rect = svgEl("rect", { x: margin.left + i * band + band * 0.2, y: rectY, width: band * 0.6, height: Math.max(rectH, 0), fill: COLORS[group] || "#777", rx: 2 });
      rect.addEventListener("mousemove", e => showTip(e, `<b>${label}</b><br>${labelFor(group)}: ${options.formatValue ? options.formatValue(value) : fmt.format(value)}`));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
      y0 += value;
    });
    const text = svgEl("text", { x: margin.left + i * band + band / 2, y: height - 12, "text-anchor": "middle", class: "tick" });
    text.textContent = label;
    svg.appendChild(text);
  });
  if (legendId) document.getElementById(legendId).innerHTML = groups.map(g => `<span><span class="swatch" style="background:${COLORS[g] || "#777"}"></span>${labelFor(g)}</span>`).join("");
}

function renderGroupedBars(svgId, rows, xKey, yKey, groups, labelKey, legendId, options = {}) {
  const svg = document.getElementById(svgId); clear(svg);
  const { width, height, margin } = dims(svg);
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const labels = [...new Set(rows.map(r => r[xKey]))].sort();
  const divisor = options.divisor || 1;
  const values = rows.map(r => (r[yKey] || 0) / divisor);
  const max = niceUpperBound(Math.max(...values, 0) * 1.12);
  const y = scale(0, max, margin.top + plotH, margin.top);
  addYAxis(svg, margin.left, y, max, margin.top, margin.top + plotH, options.formatTick || (v => fmt0.format(v)), options.axisLabel || "");
  const band = plotW / labels.length;
  const groupGap = band * 0.18;
  const innerW = band - groupGap * 2;
  const barW = innerW / groups.length;
  labels.forEach((label, i) => {
    groups.forEach((group, j) => {
      const row = rows.find(r => r[xKey] === label && r[labelKey] === group);
      const rawValue = row ? row[yKey] : 0;
      const value = rawValue / divisor;
      const x = margin.left + i * band + groupGap + j * barW + barW * 0.08;
      const rect = svgEl("rect", {
        x,
        y: y(value),
        width: barW * 0.84,
        height: y(0) - y(value),
        fill: COLORS[group] || "#777",
        rx: 2
      });
      rect.addEventListener("mousemove", e => showTip(e, `<b>${label}</b><br>${labelFor(group)}: ${options.formatValue ? options.formatValue(value) : fmt.format(value)}`));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
    });
    const text = svgEl("text", { x: margin.left + i * band + band / 2, y: height - 12, "text-anchor": "middle", class: "tick" });
    text.textContent = label;
    svg.appendChild(text);
  });
  if (legendId) document.getElementById(legendId).innerHTML = groups.map(g => `<span><span class="swatch" style="background:${COLORS[g] || "#777"}"></span>${labelFor(g)}</span>`).join("");
}

function renderAveragePrices() {
  const svg = document.getElementById("averagePriceChart"); clear(svg);
  const { width, height, margin } = dims(svg);
  const rows = (DATA.averagePrices || DATA.summary.map(row => ({
    period: row.period,
    average_price_eur_per_mwh: row.average_price
  }))).slice().sort((a, b) => Number(a.period) - Number(b.period));
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const max = niceUpperBound(Math.max(...rows.map(r => r.average_price_eur_per_mwh), 0) * 1.12);
  const y = scale(0, max, margin.top + plotH, margin.top);
  addYAxis(svg, margin.left, y, max, margin.top, margin.top + plotH, v => fmt0.format(v), "EUR/MWh");
  const band = plotW / rows.length;
  rows.forEach((row, i) => {
    const value = row.average_price_eur_per_mwh;
    const rect = svgEl("rect", {
      x: margin.left + i * band + band * 0.24,
      y: y(value),
      width: band * 0.52,
      height: y(0) - y(value),
      fill: COLORS.demand,
      rx: 2
    });
    rect.addEventListener("mousemove", e => showTip(e, `<b>${row.period}</b><br>Average price: ${fmt2.format(value)} EUR/MWh`));
    rect.addEventListener("mouseleave", hideTip);
    svg.appendChild(rect);
    const text = svgEl("text", { x: margin.left + i * band + band / 2, y: height - 12, "text-anchor": "middle", class: "tick" });
    text.textContent = row.period;
    svg.appendChild(text);
  });
}

function renderCapture() {
  const svg = document.getElementById("captureChart"); clear(svg);
  const { width, height, margin } = dims(svg);
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const periods = [...new Set(DATA.capture.map(r => r.period))].sort();
  const techs = (DATA.captureTechnologies || ["wind", "solar", "gas", "gas_turbine_cc", "demand"]).filter(tech => tech !== "gas");
  const chartRows = DATA.capture.filter(row => techs.includes(row.technology));
  const max = Math.max(...chartRows.map(r => r.capture_rate), 1) * 1.15;
  const y = scale(0, max, margin.top + plotH, margin.top);
  addYAxis(svg, margin.left, y, max, margin.top, margin.top + plotH, v => fmt2.format(v), "Capture rate");
  const band = plotW / periods.length;
  const innerW = band * 0.7;
  const barW = innerW / techs.length;
  periods.forEach((period, i) => {
    techs.forEach((tech, j) => {
      const row = DATA.capture.find(r => r.period === period && r.technology === tech);
      const value = row ? row.capture_rate : 0;
      const rect = svgEl("rect", { x: margin.left + i * band + band * 0.15 + j * barW, y: y(value), width: barW * 0.85, height: y(0) - y(value), fill: COLORS[tech], rx: 2 });
      const capturePrice = row ? row.capture_price : 0;
      rect.addEventListener("mousemove", e => showTip(e, `<b>${period} ${labelFor(tech)}</b><br>Capture rate: ${fmt2.format(value)}<br>Capture price: ${fmt2.format(capturePrice)} EUR/MWh`));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
    });
    const text = svgEl("text", { x: margin.left + i * band + band / 2, y: height - 12, "text-anchor": "middle", class: "tick" });
    text.textContent = period;
    svg.appendChild(text);
  });
  document.getElementById("captureLegend").innerHTML = techs.map(g => `<span><span class="swatch" style="background:${COLORS[g]}"></span>${labelFor(g)}</span>`).join("");
}

function renderDuration() {
  const svg = document.getElementById("durationChart"); clear(svg);
  const { width, height, margin } = dims(svg);
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const periods = Object.keys(DATA.duration || {}).sort((a, b) => Number(a) - Number(b));
  const maxHours = Math.max(...periods.map(period => (DATA.duration[period] || []).length), 1);
  const maxY = Math.max(DATA.gasMarginalCostMax, 1);
  const x = scale(1, maxHours, margin.left, margin.left + plotW);
  const y = scale(0, maxY, margin.top + plotH, margin.top);
  addYAxis(svg, margin.left, y, maxY, margin.top, margin.top + plotH, v => `${fmt0.format(v)}`, "EUR/MWh");
  periods.forEach((period, periodIndex) => {
    const rows = DATA.duration[period] || [];
    const color = YEAR_COLORS[periodIndex % YEAR_COLORS.length];
    const points = rows.map(r => ({ x: r.hour, y: Math.min(r.price_eur_per_mwh, maxY) }));
    svg.appendChild(svgEl("path", { d: pathLine(points, x, y), fill: "none", stroke: color, "stroke-width": 2.4 }));
    rows.filter((_, i) => i % 90 === 0).forEach(r => {
      const displayPrice = Math.min(r.price_eur_per_mwh, maxY);
      const capped = r.price_eur_per_mwh > maxY ? "<br>Displayed at gas marginal-cost cap" : "";
      const hit = svgEl("circle", { cx: x(r.hour), cy: y(displayPrice), r: 4, fill: "transparent" });
      hit.addEventListener("mousemove", e => showTip(e, `<b>${period}, price rank ${r.hour}</b><br>Price: ${fmt2.format(r.price_eur_per_mwh)} EUR/MWh<br>Y-axis cap: ${fmt2.format(maxY)} EUR/MWh${capped}<br>Demand at hour: ${fmt1.format(r.demand / 1000)} GW`));
      hit.addEventListener("mouseleave", hideTip);
      svg.appendChild(hit);
    });
  });
  document.getElementById("durationLegend").innerHTML = periods.map((period, i) => `<span><span class="swatch" style="background:${YEAR_COLORS[i % YEAR_COLORS.length]}"></span>${period}</span>`).join("");
}

function comparisonDatasets() {
  const climateYear = comparisonClimateYear();
  return selectedComparisonRegions()
    .map((country, i) => ({
      country,
      color: COUNTRY_COLORS[i % COUNTRY_COLORS.length],
      data: DATASETS[country]?.[climateYear]
    }))
    .filter(item => item.data);
}

function comparisonGenerationDatasets() {
  const climateYear = comparisonClimateYear();
  const items = [];
  selectedComparisonRegions().forEach(country => {
    const data = DATASETS[country]?.[climateYear];
    if (!data) return;
    if (data.regions) {
      Object.keys(data.regions).sort().forEach(region => {
        items.push({
          country: `${country} ${region}`,
          data: data.regions[region]
        });
      });
      return;
    }
    items.push({ country, data });
  });
  return items.map((item, i) => ({ ...item, color: COUNTRY_COLORS[i % COUNTRY_COLORS.length] }));
}

function renderComparisonPrices(items) {
  const svg = document.getElementById("comparisonPriceChart"); clear(svg);
  const { width, height, margin } = dims(svg);
  if (!items.length) {
    addEmpty(svg, "No selected regions have data for this climate year.");
    document.getElementById("comparisonPriceLegend").innerHTML = "";
    return;
  }
  const periods = [...new Set(items.flatMap(item => item.data.averagePrices.map(row => row.period)))]
    .sort((a, b) => Number(a) - Number(b));
  const values = items.flatMap(item => item.data.averagePrices.map(row => row.average_price_eur_per_mwh));
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const max = niceUpperBound(Math.max(...values, 1) * 1.12);
  const x = scale(0, Math.max(periods.length - 1, 1), margin.left, margin.left + plotW);
  const y = scale(0, max, margin.top + plotH, margin.top);
  addYAxis(svg, margin.left, y, max, margin.top, margin.top + plotH, v => fmt0.format(v), "EUR/MWh");
  periods.forEach((period, i) => {
    const text = svgEl("text", { x: x(i), y: height - 12, "text-anchor": "middle", class: "tick" });
    text.textContent = period;
    svg.appendChild(text);
  });
  items.forEach(item => {
    const rowsByPeriod = Object.fromEntries(item.data.averagePrices.map(row => [row.period, row]));
    const points = periods
      .map((period, i) => ({ row: rowsByPeriod[period], xIndex: i }))
      .filter(point => point.row)
      .map(point => ({ x: point.xIndex, y: point.row.average_price_eur_per_mwh, period: point.row.period }));
    svg.appendChild(svgEl("path", { d: pathLine(points, x, y), fill: "none", stroke: item.color, "stroke-width": 2.4 }));
    points.forEach(point => {
      const dot = svgEl("circle", { cx: x(point.x), cy: y(point.y), r: 4, fill: item.color });
      dot.addEventListener("mousemove", e => showTip(e, `<b>${item.country} ${point.period}</b><br>Annual baseload price: ${fmt2.format(point.y)} EUR/MWh`));
      dot.addEventListener("mouseleave", hideTip);
      svg.appendChild(dot);
    });
  });
  document.getElementById("comparisonPriceLegend").innerHTML = items.map(item => `<span><span class="swatch" style="background:${item.color}"></span>${item.country}</span>`).join("");
}

function renderComparisonGeneration(items) {
  const svg = document.getElementById("comparisonGenerationChart"); clear(svg);
  const { width, height, margin } = dims(svg);
  const carriers = ["gas_turbine_cc", "gas", "wind", "solar", "load_shedding"];
  if (!items.length) {
    addEmpty(svg, "No selected regions have data for this climate year.");
    document.getElementById("comparisonGenerationLegend").innerHTML = "";
    return;
  }
  const periods = [...new Set(items.flatMap(item => (item.data.generationShares || []).map(row => row.period)))]
    .sort((a, b) => Number(a) - Number(b));
  if (!periods.length) {
    addEmpty(svg, "No generation-share data is available.");
    document.getElementById("comparisonGenerationLegend").innerHTML = "";
    return;
  }
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const y = scale(0, 100, margin.top + plotH, margin.top);
  addYAxis(svg, margin.left, y, 100, margin.top, margin.top + plotH, v => `${fmt0.format(v)}%`, "Share");
  const periodBand = plotW / periods.length;
  const groupGap = periodBand * 0.16;
  const innerW = periodBand - groupGap * 2;
  const barW = innerW / items.length;
  periods.forEach((period, periodIndex) => {
    items.forEach((region, regionIndex) => {
      const rows = region.data.generationShares || [];
      let y0 = 0;
      carriers.forEach(carrier => {
        const row = rows.find(item => item.period === period && item.carrier === carrier);
        const value = row ? row.share : 0;
        const rectY = y(y0 + value), rectH = y(y0) - y(y0 + value);
        const x = margin.left + periodIndex * periodBand + groupGap + regionIndex * barW + barW * 0.1;
        const rect = svgEl("rect", {
          x,
          y: rectY,
          width: barW * 0.8,
          height: Math.max(rectH, 0),
          fill: COLORS[carrier],
          rx: 2
        });
        const energy = row ? row.energy_twh : 0;
        rect.addEventListener("mousemove", e => showTip(e, `<b>${region.country} ${period}</b><br>${generationShareLabel(carrier)}: ${fmt1.format(value)}%<br>${fmt2.format(energy)} TWh`));
        rect.addEventListener("mouseleave", hideTip);
        svg.appendChild(rect);
        y0 += value;
      });
      const regionText = svgEl("text", {
        x: margin.left + periodIndex * periodBand + groupGap + regionIndex * barW + barW / 2,
        y: height - 24,
        "text-anchor": "middle",
        class: "tick"
      });
      regionText.textContent = region.country;
      svg.appendChild(regionText);
    });
    const periodText = svgEl("text", {
      x: margin.left + periodIndex * periodBand + periodBand / 2,
      y: height - 8,
      "text-anchor": "middle",
      class: "tick"
    });
    periodText.textContent = period;
    svg.appendChild(periodText);
  });
  document.getElementById("comparisonGenerationLegend").innerHTML = carriers.map(carrier => `<span><span class="swatch" style="background:${COLORS[carrier]}"></span>${generationShareLabel(carrier)}</span>`).join("");
}

function renderComparison() {
  renderComparisonPrices(comparisonDatasets());
  renderComparisonGeneration(comparisonGenerationDatasets());
}

function selectedWelfareEffect() {
  const select = document.getElementById("welfareScenarioSelect");
  const index = Number(select.value || 0);
  return WELFARE_EFFECTS[index];
}

function renderWelfareTable(effect) {
  const container = document.getElementById("welfareSummary");
  if (!effect) {
    container.innerHTML = `<div class="empty">No forced-capacity welfare comparison is available.</div>`;
    return;
  }
  const forced = effect.forcedCapacity || {};
  const totalsByCountry = new Map();
  (effect.rows || []).filter(row => row.country !== "Total").forEach(row => {
    if (!totalsByCountry.has(row.country)) {
      totalsByCountry.set(row.country, {
        country: row.country,
        consumer_surplus_meur: 0,
        producer_surplus_meur: 0,
        congestion_rent_meur: 0,
        sew_meur: 0
      });
    }
    const total = totalsByCountry.get(row.country);
    total.consumer_surplus_meur += row.consumer_surplus_meur || 0;
    total.producer_surplus_meur += row.producer_surplus_meur || 0;
    total.congestion_rent_meur += row.congestion_rent_meur || 0;
    total.sew_meur += row.sew_meur || 0;
  });
  const tableRows = [...totalsByCountry.values()];
  if (tableRows.length) {
    tableRows.push({
      country: "Grand total",
      consumer_surplus_meur: tableRows.reduce((sum, row) => sum + row.consumer_surplus_meur, 0),
      producer_surplus_meur: tableRows.reduce((sum, row) => sum + row.producer_surplus_meur, 0),
      congestion_rent_meur: tableRows.reduce((sum, row) => sum + row.congestion_rent_meur, 0),
      sew_meur: tableRows.reduce((sum, row) => sum + row.sew_meur, 0)
    });
  }
  const rows = tableRows.map(row => `
    <tr>
      <td>${row.country}</td>
      <td>${fmt1.format(row.consumer_surplus_meur)}</td>
      <td>${fmt1.format(row.producer_surplus_meur)}</td>
      <td>${fmt1.format(row.congestion_rent_meur)}</td>
      <td>${fmt1.format(row.sew_meur)}</td>
    </tr>
  `).join("");
  container.innerHTML = `
    <p class="muted-inline">
      ${effect.scenario} vs ${effect.baseScenario}, weather year ${effect.weatherYear}.
      Forced ${forced.generator || "capacity"} from ${fmt1.format(forced.base_capacity_mw || 0)} MW
      to ${fmt1.format(forced.forced_capacity_mw || 0)} MW.
    </p>
    <table class="security-table">
      <thead>
        <tr>
          <th>Country</th>
          <th>Consumer surplus</th>
          <th>Producer surplus</th>
          <th>Congestion rent</th>
          <th>Net SEW</th>
        </tr>
        <tr>
          <th></th>
          <th>MEUR</th>
          <th>MEUR</th>
          <th>MEUR</th>
          <th>MEUR</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderWelfareChart(effect) {
  const svg = document.getElementById("welfareChart"); clear(svg);
  const legend = document.getElementById("welfareLegend");
  if (!effect || !(effect.rows || []).length) {
    addEmpty(svg, "No welfare data is available.");
    legend.innerHTML = "";
    return;
  }
  const rows = effect.rows.filter(row => row.country !== "Total");
  const components = [
    { key: "consumer_surplus_meur", colorKey: "consumer_surplus", label: "Consumer surplus" },
    { key: "producer_surplus_meur", colorKey: "producer_surplus", label: "Producer surplus" },
    { key: "congestion_rent_meur", colorKey: "congestion_rent", label: "Congestion rent" }
  ];
  const { width, height, margin } = dims(svg);
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const positives = rows.map(row => components.reduce((sum, component) => sum + Math.max(row[component.key] || 0, 0), 0));
  const negatives = rows.map(row => components.reduce((sum, component) => sum + Math.min(row[component.key] || 0, 0), 0));
  const dots = rows.map(row => row.sew_meur || 0);
  const maxY = Math.max(...positives, ...dots, 1) * 1.15;
  const minY = Math.min(...negatives, ...dots, 0) * 1.15;
  const y = scale(minY, maxY, margin.top + plotH, margin.top);
  addYAxis(svg, margin.left, y, maxY, margin.top, margin.top + plotH, v => fmt0.format(v), "MEUR");
  svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y(0), y2: y(0), stroke: "#7c8794", "stroke-width": 1.2 }));
  if (minY < 0) {
    for (let i = 1; i <= 2; i++) {
      const value = minY * i / 2;
      const text = svgEl("text", { x: margin.left - 8, y: y(value) + 4, "text-anchor": "end", class: "tick" });
      text.textContent = fmt0.format(value);
      svg.appendChild(text);
    }
  }
  const band = plotW / rows.length;
  rows.forEach((row, rowIndex) => {
    let positiveY0 = 0;
    let negativeY0 = 0;
    const xCenter = margin.left + rowIndex * band + band / 2;
    components.forEach(component => {
      const value = row[component.key] || 0;
      const base = value >= 0 ? positiveY0 : negativeY0;
      const next = base + value;
      const rect = svgEl("rect", {
        x: xCenter - band * 0.24,
        y: value >= 0 ? y(next) : y(base),
        width: band * 0.48,
        height: Math.abs(y(base) - y(next)),
        fill: COLORS[component.colorKey],
        rx: 2
      });
      rect.addEventListener("mousemove", e => showTip(e, `<b>${row.country} ${row.period}</b><br>${component.label}: ${fmt1.format(value)} MEUR`));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
      if (value >= 0) positiveY0 = next;
      else negativeY0 = next;
    });
    const dot = svgEl("circle", { cx: xCenter, cy: y(row.sew_meur || 0), r: 5, fill: COLORS.sew });
    dot.addEventListener("mousemove", e => showTip(e, `<b>${row.country} ${row.period}</b><br>Net SEW: ${fmt1.format(row.sew_meur || 0)} MEUR`));
    dot.addEventListener("mouseleave", hideTip);
    svg.appendChild(dot);
    const text = svgEl("text", { x: xCenter, y: height - 12, "text-anchor": "middle", class: "tick" });
    text.textContent = `${row.country} ${row.period}`;
    svg.appendChild(text);
  });
  legend.innerHTML = [
    ...components.map(component => `<span><span class="swatch" style="background:${COLORS[component.colorKey]}"></span>${component.label}</span>`),
    `<span><span class="swatch" style="background:${COLORS.sew};border-radius:50%"></span>Net SEW</span>`
  ].join("");
}

function renderWelfare() {
  const effect = selectedWelfareEffect();
  renderWelfareTable(effect);
  renderWelfareChart(effect);
}

function renderWeek() {
  const period = document.getElementById("weekPeriod").value;
  const isCompact = DATA.weekDataMode === "compact";
  const source = DATA.weekData[period] || (isCompact ? {} : []);
  const preset = activeWeekPreset || "shed";
  const rows = isCompact
    ? ((source[preset] || source.shed || source.winter || { rows: [] }).rows || [])
    : source.filter(r => r.timestep.slice(0, 10) >= document.getElementById("weekStart").value).slice(0, 168);
  const label = isCompact ? (source[preset]?.label || source.shed?.label || source.winter?.label || "") : "";
  document.getElementById("weekRangeLabel").textContent = label ? `Showing ${label}` : "";
  const svg = document.getElementById("weekChart"); clear(svg);
  if (!rows.length) {
    const msg = svgEl("text", { x: 70, y: 90, class: "tick" }); msg.textContent = "No data for selected week."; svg.appendChild(msg); return;
  }
  const { width, height, margin } = dims(svg);
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const carriers = DATA.dispatchCarriers;
  const positiveTotals = rows.map(r => carriers.reduce((sum, carrier) => sum + Math.max(r[carrier] || 0, 0), 0) / 1000);
  const negativeTotals = rows.map(r => carriers.reduce((sum, carrier) => sum + Math.min(r[carrier] || 0, 0), 0) / 1000);
  const maxY = Math.max(...rows.map(r => r.demand / 1000), ...positiveTotals, 1) * 1.05;
  const minY = Math.min(...negativeTotals, 0) * 1.05;
  const x = scale(0, rows.length - 1, margin.left, margin.left + plotW);
  const y = scale(minY, maxY, margin.top + plotH, margin.top);
  const axisMax = Math.max(Math.abs(minY), Math.abs(maxY));
  addYAxis(svg, margin.left, y, maxY, margin.top, margin.top + plotH, v => fmt1.format(v), "GW");
  if (minY < 0) {
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y(0), y2: y(0), stroke: "#7c8794", "stroke-width": 1.2 }));
    for (let i = 1; i <= 2; i++) {
      const value = -axisMax * i / 2;
      const text = svgEl("text", { x: margin.left - 8, y: y(value) + 4, "text-anchor": "end", class: "tick" });
      text.textContent = fmt1.format(value);
      svg.appendChild(text);
    }
  }
  const band = plotW / rows.length;
  rows.forEach((r, i) => {
    let positiveY0 = 0;
    let negativeY0 = 0;
    carriers.forEach(carrier => {
      const value = (r[carrier] || 0) / 1000;
      const base = value >= 0 ? positiveY0 : negativeY0;
      const next = base + value;
      const rectY = value >= 0 ? y(next) : y(base);
      const rectH = Math.abs(y(base) - y(next));
      const rect = svgEl("rect", { x: x(i) - band * 0.45, y: rectY, width: Math.max(band * 0.9, 1), height: Math.max(rectH, 0), fill: COLORS[carrier] });
      const dispatchLines = carriers.map(name => `${labelFor(name)}: ${fmt1.format((r[name] || 0) / 1000)} GW`).join("<br>");
      rect.addEventListener("mousemove", e => showTip(e, `<b>${period} ${r.timestep}</b><br>Demand: ${fmt1.format(r.demand / 1000)} GW<br>${dispatchLines}<br>Price: ${fmt2.format(r.price_eur_per_mwh)} EUR/MWh`));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
      if (value >= 0) positiveY0 = next;
      else negativeY0 = next;
    });
  });
  const demandLine = rows.map((r, i) => ({ x: i, y: r.demand / 1000 }));
  svg.appendChild(svgEl("path", { d: pathLine(demandLine, x, y), fill: "none", stroke: COLORS.demand, "stroke-width": 2.2 }));
  document.getElementById("weekLegend").innerHTML = [...carriers, "demand"]
    .map(carrier => `<span><span class="swatch" style="background:${COLORS[carrier]}"></span>${labelFor(carrier)}</span>`)
    .join("");
}

function clampDate(dateText) {
  if (dateText < DATA.dateMin) return DATA.dateMin;
  if (dateText > DATA.dateMax) return DATA.dateMax;
  return dateText;
}

function presetDate(monthDay) {
  const period = document.getElementById("weekPeriod").value;
  const rows = DATA.weekData[period] || [];
  const year = (rows[0]?.timestep || DATA.dateMin).slice(0, 4);
  return clampDate(`${year}-${monthDay}`);
}

function highestSheddingWeekStart(period) {
  const rows = DATA.weekData[period] || [];
  const windowSize = 168;
  if (rows.length <= windowSize) return rows[0]?.timestep.slice(0, 10) || DATA.dateMin;
  const shed = rows.map(row => Math.max(row.load_shedding || 0, 0));
  let bestEpisodeStart = -1;
  let bestEpisodeEnd = -1;
  let bestEpisodeEnergy = 0;
  let start = -1;
  let energy = 0;
  shed.forEach((value, index) => {
    if (value > 1e-6) {
      if (start < 0) start = index;
      energy += value;
      return;
    }
    if (start >= 0 && energy > bestEpisodeEnergy) {
      bestEpisodeStart = start;
      bestEpisodeEnd = index - 1;
      bestEpisodeEnergy = energy;
    }
    start = -1;
    energy = 0;
  });
  if (start >= 0 && energy > bestEpisodeEnergy) {
    bestEpisodeStart = start;
    bestEpisodeEnd = shed.length - 1;
    bestEpisodeEnergy = energy;
  }
  if (bestEpisodeEnergy <= 1e-6) return rows[0].timestep.slice(0, 10);
  let weightedIndex = 0;
  for (let i = bestEpisodeStart; i <= bestEpisodeEnd; i++) weightedIndex += i * shed[i];
  const episodeCenter = weightedIndex / bestEpisodeEnergy;
  const bestStart = Math.max(0, Math.min(Math.round(episodeCenter - (windowSize - 1) / 2), rows.length - windowSize));
  return rows[bestStart].timestep.slice(0, 10);
}

function setActiveWeekPreset(preset) {
  activeWeekPreset = preset;
  const buttons = {
    winter: document.getElementById("winterWeek"),
    summer: document.getElementById("summerWeek"),
    shed: document.getElementById("shedWeek")
  };
  Object.entries(buttons).forEach(([key, button]) => {
    const isActive = key === preset;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

function weekPresetDate(preset) {
  const period = document.getElementById("weekPeriod").value;
  if (preset === "winter") return presetDate("01-15");
  if (preset === "summer") return presetDate("07-15");
  return highestSheddingWeekStart(period);
}

function applyWeekPreset(preset) {
  setActiveWeekPreset(preset);
  if (DATA.weekDataMode !== "compact") {
    const date = document.getElementById("weekStart");
    date.value = weekPresetDate(preset);
  }
  renderWeek();
}

function initControls() {
  const periods = Object.keys(DATA.weekData).sort();
  const weekPeriod = document.getElementById("weekPeriod");
  weekPeriod.innerHTML = periods.map(p => `<option value="${p}">${p}</option>`).join("");
  weekPeriod.value = periods[0];
  const date = document.getElementById("weekStart");
  const dateLabel = document.getElementById("weekStartLabel");
  const compact = DATA.weekDataMode === "compact";
  date.hidden = compact;
  dateLabel.hidden = compact;
  setActiveWeekPreset("shed");
  if (!compact) {
    date.min = DATA.dateMin;
    date.max = DATA.dateMax;
    date.value = highestSheddingWeekStart(weekPeriod.value);
    date.onchange = () => {
      setActiveWeekPreset(null);
      renderWeek();
    };
  } else {
    date.onchange = null;
  }
  weekPeriod.onchange = () => applyWeekPreset(activeWeekPreset || "shed");
  document.getElementById("winterWeek").onclick = () => applyWeekPreset("winter");
  document.getElementById("summerWeek").onclick = () => applyWeekPreset("summer");
  document.getElementById("shedWeek").onclick = () => applyWeekPreset("shed");
}

function renderAll() {
  renderStackedBars(
    "capacityChart",
    DATA.activeCapacity,
    "period",
    "p_nom_opt_mw",
    DATA.capacityCarriers,
    "carrier",
    "capacityLegend",
    {
      divisor: 1000,
      axisLabel: "GW",
      formatTick: v => fmt1.format(v),
      formatValue: v => `${fmt1.format(v)} GW`
    }
  );
  renderStackedBars(
    "buildCapacityChart",
    DATA.buildCapacity,
    "build_year",
    "p_nom_opt_mw",
    DATA.capacityCarriers,
    "carrier",
    "buildCapacityLegend",
    {
      divisor: 1000,
      axisLabel: "GW",
      formatTick: v => fmt1.format(v),
      formatValue: v => `${fmt1.format(v)} GW`
    }
  );
  renderAveragePrices();
  renderCapture();
  renderSecurity();
  renderWeek();
  renderDuration();
  renderGroupedBars(
    "capexChart",
    DATA.buildCapex,
    "build_year",
    "capex_meur_per_mw",
    DATA.capacityCarriers,
    "carrier",
    "capexLegend",
    {
      axisLabel: "MEUR/MW",
      formatTick: v => fmt2.format(v),
      formatValue: v => `${fmt2.format(v)} MEUR/MW`
    }
  );
}

initTabs();
initCountryControl();
initComparisonControls();
initWelfareControls();
initOutageControls();
initControls();
renderModelSummary();
renderAll();
window.addEventListener("resize", () => {
  renderAll();
  if (!document.getElementById("comparePanel").hidden) renderComparison();
  if (!document.getElementById("outagePanel").hidden) renderOutageComparison();
  if (!document.getElementById("welfarePanel").hidden) renderWelfare();
});
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the PyPSA output dashboard HTML.")
    parser.add_argument(
        "--full-week-data",
        action="store_true",
        help="Embed all hourly dispatch rows so the Dispatch Week datepicker can select any week.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dashboard_bundle(OUTPUT_DIR, compact_week_data=not args.full_week_data)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote dashboard to {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
