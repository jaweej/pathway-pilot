from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pathway_pilot.config import DEV_DATA_DIR


OUTPUT_DIR = DEV_DATA_DIR / "pathway-pilot" / "output"
DASHBOARD_PATH = OUTPUT_DIR / "pypsa_output_dashboard.html"
CAPACITY_CARRIERS = ["gas_turbine_cc", "gas", "wind", "solar"]
DISPATCH_CARRIERS = [*CAPACITY_CARRIERS, "load_shedding"]
GAS_CARRIERS = ["gas", "gas_turbine_cc"]


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


def _dashboard_payload(
    capacities: pd.DataFrame,
    dispatch: pd.DataFrame,
    prices: pd.DataFrame,
    metadata: dict,
    region: str | None = None,
    flows: pd.DataFrame | None = None,
) -> dict:
    regions = set((metadata.get("model_regions") or {}).keys())
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
    dispatch_carriers = [*DISPATCH_CARRIERS]
    if region is not None and imports is not None:
        dispatch_carriers.extend(["interconnector_import", "interconnector_export"])

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

    week_data = {}
    for period, group in hourly.groupby("period"):
        ordered = group.sort_values("timestep")
        week_data[str(period)] = _series_points(
            ordered,
            [
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
            ],
        )

    summary = []
    security = []
    generation_shares = []
    for period, group in hourly.groupby("period"):
        shed = group["load_shedding"].clip(lower=0)
        shed_hours = int((shed > 1e-6).sum())
        shed_energy_mwh = float(shed.sum())
        demand_energy_mwh = float(group["demand"].sum())
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
        "dateMin": min(row["timestep"][:10] for rows in week_data.values() for row in rows),
        "dateMax": max(row["timestep"][:10] for rows in week_data.values() for row in rows),
        "metadata": {**metadata, "selected_region": region},
    }


def load_dashboard_data(output_dir: Path = OUTPUT_DIR) -> dict:
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
    data = _dashboard_payload(capacities, dispatch, prices, metadata, flows=flows)
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


def load_dashboard_bundle(output_dir: Path = OUTPUT_DIR) -> dict:
    datasets = {}

    def add_dataset(scenario_dir: Path) -> None:
        data = load_dashboard_data(scenario_dir)
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

    preferred = "NL" if "NL" in datasets else sorted(datasets)[0]
    preferred_climate = sorted(datasets[preferred], key=int)[0]
    return {
        "datasets": datasets,
        "defaultCountry": preferred,
        "defaultClimateYear": preferred_climate,
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
      <h2>Active Capacity By Model Year</h2>
      <svg id="capacityChart"></svg>
      <div class="legend" id="capacityLegend"></div>
    </div>
    <div class="panel">
      <h2>Capacity Built By Investment Year</h2>
      <svg id="buildCapacityChart"></svg>
      <div class="legend" id="buildCapacityLegend"></div>
    </div>
    <div class="panel">
      <h2>Average Prices</h2>
      <svg id="averagePriceChart"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:var(--demand)"></span>Average electricity price</span>
      </div>
    </div>
    <div class="panel">
      <h2>Duration Curve</h2>
      <svg id="durationChart"></svg>
      <div class="legend" id="durationLegend"></div>
    </div>
    <div class="panel wide">
      <h2>Dispatch Week</h2>
      <div class="controls">
        <label for="weekPeriod">Model year</label>
        <select id="weekPeriod"></select>
        <label for="weekStart">Week start</label>
        <input id="weekStart" type="date">
        <button id="winterWeek" type="button">Winter week</button>
        <button id="summerWeek" type="button">Summer week</button>
        <button id="shedWeek" type="button">Highest shedding</button>
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
      <h2>Security Of Supply</h2>
      <div id="securityTable"></div>
    </div>
    <div class="panel">
      <h2>Capture Rates</h2>
      <svg id="captureChart"></svg>
      <div class="legend" id="captureLegend"></div>
    </div>
    <div class="panel">
      <h2>Unit CAPEX By Investment Year</h2>
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
        <h2>Annual Baseload Prices</h2>
        <svg id="comparisonPriceChart"></svg>
        <div class="legend" id="comparisonPriceLegend"></div>
      </div>
      <div class="panel">
        <h2>Generation Shares</h2>
        <svg id="comparisonGenerationChart"></svg>
        <div class="legend" id="comparisonGenerationLegend"></div>
      </div>
    </div>
  </section>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const APP_DATA = __DATA__;
const DATASETS = APP_DATA.datasets || {};
let DATA;
let activeWeekPreset = "shed";
const COLORS = { wind: "#2f80ed", solar: "#f2b705", gas: "#7a5195", gas_turbine_cc: "#00a6a6", load_shedding: "#c43d3d", demand: "#111827", residual_load: "#18a058", interconnector_import: "#38a3a5", interconnector_export: "#ef8354" };
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
  select.value = preferredYear && years.includes(String(preferredYear)) ? String(preferredYear) : years[0];
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
    : climateYears[0];
  climateSelect.addEventListener("change", renderComparison);
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
    { button: document.getElementById("compareTab"), panel: document.getElementById("comparePanel") }
  ];
  tabs.forEach(tab => {
    tab.button.addEventListener("click", () => {
      tabs.forEach(item => {
        const selected = item === tab;
        item.button.setAttribute("aria-selected", selected ? "true" : "false");
        item.panel.hidden = !selected;
      });
      if (tab.panel.id === "comparePanel") renderComparison();
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
    voll: "VoLL (Value of Lost Load): penalty cost assigned to involuntary load shedding in the optimisation, expressed in EUR/MWh."
  };
  const rows = DATA.security.map(row => `
    <tr>
      <td>${row.period}</td>
      <td>${fmt1.format((DATA.summary.find(item => item.period === row.period)?.peak_demand_mw || 0) / 1000)} GW</td>
      <td>${fmt.format(DATA.summary.find(item => item.period === row.period)?.energy_twh || 0)} TWh</td>
      <td><span class="metric-tip" data-tip="${definitions.lole}">${fmt0.format(row.lole_hours)} h/y</span></td>
      <td><span class="metric-tip" data-tip="${definitions.eens}">${fmt2.format(row.eens_gwh)} GWh/y</span></td>
      <td><span class="metric-tip" data-tip="${definitions.eensPct}">${fmt2.format(row.eens_pct_demand)}%</span></td>
      <td><span class="metric-tip" data-tip="${definitions.peak}">${fmt1.format(row.peak_shed_mw / 1000)} GW</span></td>
      <td><span class="metric-tip" data-tip="${definitions.gas}">${fmt2.format(row.gas_generation_twh)} TWh</span></td>
      <td><span class="metric-tip" data-tip="${definitions.voll}">${fmt0.format(row.voll_eur_per_mwh)} EUR/MWh</span></td>
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
          <th><span class="metric-tip" data-tip="${definitions.voll}">VoLL</span></th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
  document.querySelectorAll(".metric-tip").forEach(el => {
    el.addEventListener("mousemove", e => showTip(e, el.dataset.tip));
    el.addEventListener("mouseleave", hideTip);
  });
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
  const items = comparisonDatasets();
  renderComparisonPrices(items);
  renderComparisonGeneration(items);
}

function renderWeek() {
  const period = document.getElementById("weekPeriod").value;
  const start = document.getElementById("weekStart").value;
  const rows = (DATA.weekData[period] || []).filter(r => r.timestep.slice(0, 10) >= start).slice(0, 168);
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
  let running = shed.slice(0, windowSize).reduce((sum, value) => sum + value, 0);
  const sums = [running];
  for (let i = windowSize; i < shed.length; i++) {
    running += shed[i] - shed[i - windowSize];
    sums.push(running);
  }
  const maxShed = Math.max(...sums);
  if (maxShed <= 1e-6) return rows[0].timestep.slice(0, 10);
  let bestStart = 0;
  let bestCenterDistance = Number.POSITIVE_INFINITY;
  sums.forEach((sum, start) => {
    if (Math.abs(sum - maxShed) > 1e-6) return;
    let weightedIndex = 0;
    for (let i = start; i < start + windowSize; i++) weightedIndex += i * shed[i];
    const sheddingCenter = weightedIndex / sum;
    const windowCenter = start + (windowSize - 1) / 2;
    const centerDistance = Math.abs(sheddingCenter - windowCenter);
    if (centerDistance < bestCenterDistance) {
      bestCenterDistance = centerDistance;
      bestStart = start;
    }
  });
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
  const date = document.getElementById("weekStart");
  date.value = weekPresetDate(preset);
  setActiveWeekPreset(preset);
  renderWeek();
}

function initControls() {
  const periods = Object.keys(DATA.weekData).sort();
  const weekPeriod = document.getElementById("weekPeriod");
  weekPeriod.innerHTML = periods.map(p => `<option value="${p}">${p}</option>`).join("");
  weekPeriod.value = periods[0];
  const date = document.getElementById("weekStart");
  date.min = DATA.dateMin;
  date.max = DATA.dateMax;
  date.value = highestSheddingWeekStart(weekPeriod.value);
  setActiveWeekPreset("shed");
  weekPeriod.onchange = () => applyWeekPreset(activeWeekPreset || "shed");
  date.onchange = () => {
    setActiveWeekPreset(null);
    renderWeek();
  };
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
initControls();
renderModelSummary();
renderAll();
window.addEventListener("resize", () => {
  renderAll();
  if (!document.getElementById("comparePanel").hidden) renderComparison();
});
</script>
</body>
</html>
"""


def main() -> None:
    data = load_dashboard_bundle(OUTPUT_DIR)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote dashboard to {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
