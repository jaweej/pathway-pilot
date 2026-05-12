from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pathway_pilot.config import DEV_DATA_DIR


OUTPUT_DIR = DEV_DATA_DIR / "pathway-pilot" / "output"
DASHBOARD_PATH = OUTPUT_DIR / "pypsa_output_dashboard.html"
CAPACITY_CARRIERS = ["wind", "solar", "gas", "gas_turbine_cc"]
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


def load_dashboard_data(output_dir: Path = OUTPUT_DIR) -> dict:
    capacities = pd.read_parquet(output_dir / "optimal_capacities.parquet")
    dispatch = pd.read_parquet(output_dir / "hourly_dispatch.parquet")
    prices = pd.read_parquet(output_dir / "hourly_prices.parquet")

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
    dispatch_wide["demand"] = dispatch_wide[DISPATCH_CARRIERS].sum(axis=1)

    prices = prices[prices["bus"] == "electricity"].drop(columns=["bus"])
    hourly = dispatch_wide.merge(prices, on=["period", "timestep"], how="left")

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
                "price_eur_per_mwh",
            ],
        )

    summary = []
    security = []
    for period, group in hourly.groupby("period"):
        shed = group["load_shedding"].clip(lower=0)
        shed_hours = int((shed > 1e-6).sum())
        shed_energy_mwh = float(shed.sum())
        summary.append(
            {
                "period": int(period),
                "peak_demand_mw": _round(group["demand"].max()),
                "energy_twh": _round(group["demand"].sum() / 1_000_000),
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
                "peak_shed_mw": _round(shed.max(), 4),
                "voll_eur_per_mwh": _round(
                    capacities.loc[
                        capacities["generator"] == "load_shedding", "marginal_cost"
                    ].iloc[0],
                    2,
                ),
            }
        )

    return {
        "summary": summary,
        "buildCapacity": _series_points(build_capacity, ["build_year", "carrier", "p_nom_opt_mw"]),
        "buildCapex": _series_points(build_capex, ["build_year", "carrier", "capex_meur_per_mw"]),
        "activeCapacity": active_rows,
        "duration": duration,
        "capture": capture,
        "security": security,
        "gasMarginalCostMax": _round(gas_marginal_cost_max),
        "capacityCarriers": CAPACITY_CARRIERS,
        "captureTechnologies": [*CAPACITY_CARRIERS, "demand"],
        "dispatchCarriers": DISPATCH_CARRIERS,
        "weekData": week_data,
        "dateMin": min(row["timestep"][:10] for rows in week_data.values() for row in rows),
        "dateMax": max(row["timestep"][:10] for rows in week_data.values() for row in rows),
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
    label {
      color: var(--muted);
      font-size: 13px;
    }
    input, select {
      border: 1px solid var(--line);
      background: white;
      border-radius: 6px;
      color: var(--ink);
      padding: 7px 9px;
      font: inherit;
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
  <p>One-zone capacity expansion and dispatch results. Capacity factors use DKW1; demand is DKE1 + DKW1. The 2050 time series reuses 2040 profiles.</p>
</header>
<main>
  <section class="grid">
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
      <h2>Capture Rates</h2>
      <svg id="captureChart"></svg>
      <div class="legend" id="captureLegend"></div>
    </div>
    <div class="panel">
      <h2>Security Of Supply</h2>
      <div id="securityTable"></div>
    </div>
    <div class="panel wide">
      <h2>Dispatch Week</h2>
      <div class="controls">
        <label for="weekPeriod">Model year</label>
        <select id="weekPeriod"></select>
        <label for="weekStart">Week start</label>
        <input id="weekStart" type="date">
      </div>
      <svg id="weekChart"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:var(--wind)"></span>Wind</span>
        <span><span class="swatch" style="background:var(--solar)"></span>Solar</span>
        <span><span class="swatch" style="background:var(--gas)"></span>Gas turbine</span>
        <span><span class="swatch" style="background:var(--gas-cc)"></span>Gas turbine CC</span>
        <span><span class="swatch" style="background:var(--shed)"></span>Load shedding</span>
        <span><span class="swatch" style="background:var(--demand)"></span>Demand</span>
      </div>
    </div>
    <div class="panel wide">
      <h2>Duration Curve</h2>
      <div class="controls">
        <label for="durationPeriod">Model year</label>
        <select id="durationPeriod"></select>
      </div>
      <svg id="durationChart"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:var(--demand)"></span>Electricity price</span>
      </div>
    </div>
    <div class="panel wide">
      <h2>Unit CAPEX By Investment Year</h2>
      <svg id="capexChart"></svg>
      <div class="legend" id="capexLegend"></div>
    </div>
  </section>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = __DATA__;
const COLORS = { wind: "#2f80ed", solar: "#f2b705", gas: "#7a5195", gas_turbine_cc: "#00a6a6", load_shedding: "#c43d3d", demand: "#111827", residual_load: "#18a058" };
const LABELS = { wind: "Wind", solar: "Solar", gas: "Gas turbine", gas_turbine_cc: "Gas turbine CC", load_shedding: "Load shedding", demand: "Demand" };
const tooltip = document.getElementById("tooltip");
const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const fmt0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const fmt1 = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const fmt2 = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
function labelFor(key) { return LABELS[key] || key.replaceAll("_", " "); }

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

function renderSecurity() {
  const definitions = {
    lole: "LOLE (Loss of Load Expectation): number of hours in the simulated year with non-zero load shedding. This deterministic model-year value is reported as an LOLE-style metric.",
    eens: "EENS (Expected Energy Not Served): total unserved electricity demand in the simulated year. This deterministic model-year value is reported as an EENS-style metric.",
    peak: "Peak shed: maximum hourly load shedding in the simulated year.",
    voll: "VoLL (Value of Lost Load): penalty cost assigned to involuntary load shedding in the optimisation, expressed in EUR/MWh."
  };
  const rows = DATA.security.map(row => `
    <tr>
      <td>${row.period}</td>
      <td>${fmt1.format((DATA.summary.find(item => item.period === row.period)?.peak_demand_mw || 0) / 1000)} GW</td>
      <td>${fmt.format(DATA.summary.find(item => item.period === row.period)?.energy_twh || 0)} TWh</td>
      <td><span class="metric-tip" data-tip="${definitions.lole}">${fmt0.format(row.lole_hours)} h/y</span></td>
      <td><span class="metric-tip" data-tip="${definitions.eens}">${fmt2.format(row.eens_gwh)} GWh/y</span></td>
      <td><span class="metric-tip" data-tip="${definitions.peak}">${fmt1.format(row.peak_shed_mw / 1000)} GW</span></td>
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
          <th><span class="metric-tip" data-tip="${definitions.peak}">Peak shed</span></th>
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

function renderCapture() {
  const svg = document.getElementById("captureChart"); clear(svg);
  const { width, height, margin } = dims(svg);
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const periods = [...new Set(DATA.capture.map(r => r.period))].sort();
  const techs = DATA.captureTechnologies || ["wind", "solar", "gas", "gas_turbine_cc", "demand"];
  const max = Math.max(...DATA.capture.map(r => r.capture_rate), 1) * 1.15;
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
  const period = document.getElementById("durationPeriod").value;
  const rows = DATA.duration[period] || [];
  const svg = document.getElementById("durationChart"); clear(svg);
  const { width, height, margin } = dims(svg);
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const maxY = Math.max(DATA.gasMarginalCostMax, 1);
  const x = scale(1, rows.length, margin.left, margin.left + plotW);
  const y = scale(0, maxY, margin.top + plotH, margin.top);
  addYAxis(svg, margin.left, y, maxY, margin.top, margin.top + plotH, v => `${fmt0.format(v)}`, "EUR/MWh");
  const points = rows.map(r => ({ x: r.hour, y: Math.min(r.price_eur_per_mwh, maxY) }));
  svg.appendChild(svgEl("path", { d: pathLine(points, x, y), fill: "none", stroke: COLORS.demand, "stroke-width": 2.4 }));
  rows.filter((_, i) => i % 90 === 0).forEach(r => {
    const displayPrice = Math.min(r.price_eur_per_mwh, maxY);
    const capped = r.price_eur_per_mwh > maxY ? "<br>Displayed at gas marginal-cost cap" : "";
    const hit = svgEl("circle", { cx: x(r.hour), cy: y(displayPrice), r: 4, fill: "transparent" });
    hit.addEventListener("mousemove", e => showTip(e, `<b>${period}, price rank ${r.hour}</b><br>Price: ${fmt2.format(r.price_eur_per_mwh)} EUR/MWh<br>Y-axis cap: ${fmt2.format(maxY)} EUR/MWh${capped}<br>Demand at hour: ${fmt1.format(r.demand / 1000)} GW`));
    hit.addEventListener("mouseleave", hideTip);
    svg.appendChild(hit);
  });
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
  const maxY = Math.max(...rows.map(r => Math.max(r.demand, DATA.dispatchCarriers.reduce((sum, carrier) => sum + r[carrier], 0)) / 1000), 1) * 1.05;
  const x = scale(0, rows.length - 1, margin.left, margin.left + plotW);
  const y = scale(0, maxY, margin.top + plotH, margin.top);
  addYAxis(svg, margin.left, y, maxY, margin.top, margin.top + plotH, v => fmt1.format(v), "GW");
  const carriers = DATA.dispatchCarriers;
  const band = plotW / rows.length;
  rows.forEach((r, i) => {
    let y0 = 0;
    carriers.forEach(carrier => {
      const value = r[carrier] / 1000;
      const rectY = y(y0 + value);
      const rectH = y(y0) - y(y0 + value);
      const rect = svgEl("rect", { x: x(i) - band * 0.45, y: rectY, width: Math.max(band * 0.9, 1), height: Math.max(rectH, 0), fill: COLORS[carrier] });
      const dispatchLines = carriers.map(name => `${labelFor(name)}: ${fmt1.format(r[name] / 1000)} GW`).join("<br>");
      rect.addEventListener("mousemove", e => showTip(e, `<b>${period} ${r.timestep}</b><br>Demand: ${fmt1.format(r.demand / 1000)} GW<br>${dispatchLines}<br>Price: ${fmt2.format(r.price_eur_per_mwh)} EUR/MWh`));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
      y0 += value;
    });
  });
  const demandLine = rows.map((r, i) => ({ x: i, y: r.demand / 1000 }));
  svg.appendChild(svgEl("path", { d: pathLine(demandLine, x, y), fill: "none", stroke: COLORS.demand, "stroke-width": 2.2 }));
}

function initControls() {
  const periods = Object.keys(DATA.weekData).sort();
  for (const id of ["durationPeriod", "weekPeriod"]) {
    const select = document.getElementById(id);
    select.innerHTML = periods.map(p => `<option value="${p}">${p}</option>`).join("");
    select.value = periods[0];
  }
  const date = document.getElementById("weekStart");
  date.min = DATA.dateMin;
  date.max = DATA.dateMax;
  date.value = DATA.dateMin;
  document.getElementById("durationPeriod").addEventListener("change", renderDuration);
  document.getElementById("weekPeriod").addEventListener("change", renderWeek);
  date.addEventListener("change", renderWeek);
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

initControls();
renderAll();
window.addEventListener("resize", renderAll);
</script>
</body>
</html>
"""


def main() -> None:
    data = load_dashboard_data(OUTPUT_DIR)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote dashboard to {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
