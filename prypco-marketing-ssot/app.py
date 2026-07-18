"""
PRYPCO Blocks -- Marketing SSOT Dashboard (Streamlit + Plotly)
==============================================================
An interactive Single-Source-of-Truth dashboard that reads the staging tables
produced by data_pipeline.py and presents boardroom-level acquisition, spend
efficiency, funnel velocity, and ROAS views, plus an AI CoPilot container that
serializes the current dashboard state for a Claude/MCP executive summary.

Run:
    pip install streamlit plotly pandas
    python3 data_pipeline.py          # produces prypco_pipeline/staging/*.csv
    streamlit run app.py

Data contract (aliased to the names in the brief):
    df_acquisition       <- staging_acquisition_conversion            (channel x month x vertical)
    df_spend_efficiency  <- staging_spend_efficiency                  (channel x month)
    df_funnel_velocity   <- funnel_by_channel  +  days_to_convert     (channel grain)
    df_roas              <- staging_vertical_roas                     (channel x month x vertical)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise divide that yields NaN (not inf) on a zero denominator."""
    return numerator / denominator.replace(0, np.nan)

# ==========================================================================
# 1. CONFIG  --  paths, palette, canonical dimension orders
# ==========================================================================

STAGING_DIR = Path(__file__).parent / "staging"

VERTICALS = ["Primary Sales", "Secondary Sales", "Rental", "Mortgage", "Property Management"]

# Validated categorical palette (see dataviz reference). Color follows the
# ENTITY (vertical), never its rank, so filtering never repaints survivors.
# Five verticals map to the first five CVD-optimized slots, in fixed order.
VERTICAL_COLOR = {
    "Primary Sales":       "#2a78d6",  # slot 1 blue
    "Secondary Sales":     "#1baf7a",  # slot 2 aqua
    "Rental":              "#eda100",  # slot 3 yellow
    "Mortgage":            "#008300",  # slot 4 green
    "Property Management": "#4a3aa7",  # slot 5 violet
    "Unallocated":         "#898781",  # muted -- spend with no attributed revenue
}

# Cost metrics share one unit (AED), so one axis is legitimate. Three slots.
METRIC_COLOR = {"CAC": "#2a78d6", "CPL": "#1baf7a", "CPQL": "#eda100"}

# Funnel is ORDINAL -- one blue hue, light (top of funnel) to dark (Won).
# Steps start no lighter than sequential step 250 to clear 2:1 on light.
FUNNEL_STAGE_COLOR = {
    "Lead":       "#86b6ef",
    "Qualified":  "#3987e5",
    "Site Visit": "#256abf",
    "Won":        "#104281",
}
SINGLE_HUE = "#2a78d6"  # single-series marks (days-to-convert)

# Ink / chrome tokens
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

PLOTLY_FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                   color=INK_SECONDARY, size=13)


# ==========================================================================
# 2. PAGE SHELL + GLOBAL CSS  --  elite spacing, KPI card styling
# ==========================================================================

st.set_page_config(
    page_title="PRYPCO Blocks — Marketing SSOT",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink-1:#0b0b0b; --ink-2:#52514e; --ink-3:#898781;
              --surface:#fcfcfb; --line:#e1e0d9; --brand:#2a78d6; }

      html, body, [class*="css"] {
          font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }

      /* tighten the default Streamlit top padding for a denser boardroom view */
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }

      /* section headers */
      .section-title { font-size: 1.05rem; font-weight: 700; letter-spacing:.01em;
          color: var(--ink-1); margin: 1.9rem 0 .2rem 0; }
      .section-sub { font-size: .82rem; color: var(--ink-3); margin: 0 0 .9rem 0; }

      hr.rule { border:none; border-top:1px solid var(--line); margin:.4rem 0 1.4rem 0; }

      /* KPI metric cards */
      div[data-testid="stMetric"] {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 1.15rem 1.3rem;
          box-shadow: 0 1px 2px rgba(11,11,11,.04);
      }
      div[data-testid="stMetric"] label p {
          font-size: .78rem !important; font-weight:600; color: var(--ink-3);
          text-transform: uppercase; letter-spacing:.05em; }
      div[data-testid="stMetricValue"] {
          font-size: 1.9rem !important; font-weight:700; color: var(--ink-1);
          font-variant-numeric: tabular-nums; }

      /* AI CoPilot container */
      .copilot-shell {
          background: linear-gradient(180deg,#f4f8fe 0%, #fbfdff 100%);
          border: 1px solid #d5e4f7; border-radius: 16px;
          padding: 1.4rem 1.6rem; margin-top:.4rem; }
      .copilot-badge {
          display:inline-flex; align-items:center; gap:.5rem;
          background:#2a78d6; color:#fff; font-size:.72rem; font-weight:700;
          padding:.28rem .7rem; border-radius:999px; letter-spacing:.04em;
          text-transform:uppercase; }
      .copilot-meta { font-size:.75rem; color:var(--ink-3); margin-top:.7rem; }

      .app-title { font-size:1.7rem; font-weight:800; color:var(--ink-1); margin-bottom:.1rem; }
      .app-tag  { font-size:.9rem; color:var(--ink-2); margin-bottom:.2rem; }
      section[data-testid="stSidebar"] { border-right:1px solid var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="rule"/>', unsafe_allow_html=True)


# ==========================================================================
# 3. DATA LOADING  --  cached; aliases staging tables to the brief's df names
# ==========================================================================

@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    def _read(name: str) -> pd.DataFrame:
        path = STAGING_DIR / f"{name}.csv"
        if not path.exists():
            st.error(f"Missing staging file: {path.name}. Run `python3 data_pipeline.py` first.")
            st.stop()
        return pd.read_csv(path)

    df_acquisition = _read("staging_acquisition_conversion").rename(columns={"signup_month": "month"})
    df_spend_efficiency = _read("staging_spend_efficiency")
    df_roas = _read("staging_vertical_roas")

    # df_funnel_velocity: merge the two channel-grain tables into one, per the brief
    funnel = _read("staging_funnel_by_channel")
    velocity = _read("staging_days_to_convert_by_channel")[["channel", "mean_days", "median_days",
                                                            "p25_days", "p75_days", "p90_days"]]
    df_funnel_velocity = funnel.merge(velocity, on="channel", how="left")

    return {
        "df_acquisition": df_acquisition,
        "df_spend_efficiency": df_spend_efficiency,
        "df_funnel_velocity": df_funnel_velocity,
        "df_roas": df_roas,
    }


data = load_data()
df_acquisition = data["df_acquisition"]
df_spend_efficiency = data["df_spend_efficiency"]
df_funnel_velocity = data["df_funnel_velocity"]
df_roas = data["df_roas"]

ALL_MONTHS = sorted(set(df_roas["month"].dropna()) | set(df_spend_efficiency["month"].dropna()))
ALL_CHANNELS = sorted(df_funnel_velocity["channel"].dropna().unique())


# ==========================================================================
# 4. GLOBAL SIDEBAR FILTERS
# ==========================================================================

def _resolve(selection: list[str], universe: list[str]) -> list[str]:
    """Empty box or an explicit 'All' pick expands to the full universe;
    otherwise the chosen subset is used verbatim (multi-select supported)."""
    if not selection or "All" in selection:
        return universe
    return [s for s in universe if s in selection]  # keep canonical order


def _label(selection: list[str], universe: list[str]) -> str:
    if not selection or "All" in selection or len(selection) == len(universe):
        return "All"
    if len(selection) <= 2:
        return ", ".join(selection)
    return f"{len(selection)} selected"


with st.sidebar:
    st.markdown('<div class="app-title" style="font-size:1.2rem;">◼ PRYPCO SSOT</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="app-tag">Marketing analytics control plane</div>', unsafe_allow_html=True)
    st.markdown('<hr class="rule"/>', unsafe_allow_html=True)

    st.markdown("**Vertical**")
    vertical_pick = st.multiselect(
        "Vertical", options=["All"] + VERTICALS, default=["All"],
        label_visibility="collapsed", placeholder="All verticals",
    )
    sel_verticals = _resolve(vertical_pick, VERTICALS)
    vertical_label = _label(vertical_pick, VERTICALS)

    st.markdown("**Date range (month)**")
    if len(ALL_MONTHS) >= 2:
        month_start, month_end = st.select_slider(
            "Month range", options=ALL_MONTHS, value=(ALL_MONTHS[0], ALL_MONTHS[-1]),
            label_visibility="collapsed",
        )
    else:
        month_start, month_end = ALL_MONTHS[0], ALL_MONTHS[-1]
    sel_months = [m for m in ALL_MONTHS if month_start <= m <= month_end]

    st.markdown("**Channel**")
    channel_pick = st.multiselect(
        "Channel", options=["All"] + ALL_CHANNELS, default=["All"],
        label_visibility="collapsed", placeholder="All channels",
    )
    sel_channels = _resolve(channel_pick, ALL_CHANNELS)
    channel_label = _label(channel_pick, ALL_CHANNELS)



# ---- filter application -------------------------------------------------

def _mask_months(df: pd.DataFrame) -> pd.Series:
    return df["month"].isin(sel_months)


f_acq = df_acquisition[
    df_acquisition["vertical"].isin(sel_verticals)
    & _mask_months(df_acquisition)
    & df_acquisition["channel"].isin(sel_channels)
].copy()

f_spend = df_spend_efficiency[
    _mask_months(df_spend_efficiency) & df_spend_efficiency["channel"].isin(sel_channels)
].copy()

f_roas = df_roas[
    df_roas["vertical"].isin(sel_verticals + ["Unallocated"])
    & _mask_months(df_roas)
    & df_roas["channel"].isin(sel_channels)
].copy()

f_funnel = df_funnel_velocity[df_funnel_velocity["channel"].isin(sel_channels)].copy()


# ==========================================================================
# 5. HEADER + KPI CARDS
# ==========================================================================

st.markdown('<div class="app-title">Marketing Single-Source-of-Truth</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="app-tag">PRYPCO Blocks &nbsp;·&nbsp; {month_start} → {month_end} '
    f'&nbsp;·&nbsp; Vertical: {vertical_label} &nbsp;·&nbsp; Channel: {channel_label}</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="rule"/>', unsafe_allow_html=True)


def _fmt_aed(x: float, big: bool = True) -> str:
    if x is None or pd.isna(x):
        return "—"
    if big and abs(x) >= 1e9:
        return f"AED {x/1e9:.2f}B"
    if big and abs(x) >= 1e6:
        return f"AED {x/1e6:.1f}M"
    if big and abs(x) >= 1e3:
        return f"AED {x/1e3:.1f}K"
    return f"AED {x:,.0f}"


# realized revenue & spend are read off the internally-consistent ROAS table
total_revenue = f_roas["revenue_aed"].sum()
total_alloc_spend = f_roas["allocated_spend_aed"].sum()
overall_roas = total_revenue / total_alloc_spend if total_alloc_spend else float("nan")

# blended CAC uses the spend table (spend / converted customers)
total_spend = f_spend["actual_spend_aed"].sum()
total_converted = f_spend["crm_converted_customers"].sum()
blended_cac = total_spend / total_converted if total_converted else float("nan")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue (AED)", _fmt_aed(total_revenue))
k2.metric("Total Ad Spend (AED)", _fmt_aed(total_spend))
k3.metric("Blended CAC", _fmt_aed(blended_cac, big=False) if pd.notna(blended_cac) else "—")
k4.metric("Overall ROAS", f"{overall_roas:,.0f}×" if pd.notna(overall_roas) else "—")

st.caption(
    "ROAS is intentionally shown as a ratio, not a headline of health: PRYPCO ticket "
    "sizes (AED millions) dwarf digital spend, so a high multiple reflects deal economics "
    "more than channel efficiency. Read CAC and cost-per-qualified-lead alongside it."
)


# ==========================================================================
# 6. VISUALIZATIONS
# ==========================================================================

def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, font=PLOTLY_FONT, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        margin=dict(l=10, r=16, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12)),
        hoverlabel=dict(font_size=12, font_family=PLOTLY_FONT["family"]),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=INK_MUTED, size=11))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(color=INK_MUTED, size=11))
    return fig


# ---- 6A. Spend efficiency trends -----------------------------------------
# CAC sits 1-2 orders of magnitude above CPL/CPQL, so they get separate
# charts (each with a single AED scale) rather than one axis where the smaller
# two flatten to the baseline. CPL and CPQL share a chart -- same unit, same
# order of magnitude, and the gap between them IS the qualification premium.

def _cost_agg(spend: pd.DataFrame) -> pd.DataFrame:
    agg = (spend.groupby("month", as_index=False)
           .agg(spend=("actual_spend_aed", "sum"),
                converted=("crm_converted_customers", "sum"),
                leads=("leads_generated", "sum"),
                qualified=("qualified_leads", "sum")))
    agg["CAC"] = _ratio(agg["spend"], agg["converted"])
    agg["CPL"] = _ratio(agg["spend"], agg["leads"])
    agg["CPQL"] = _ratio(agg["spend"], agg["qualified"])
    return agg


def chart_cac_trend(spend: pd.DataFrame) -> go.Figure:
    agg = _cost_agg(spend)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["month"], y=agg["CAC"], name="Blended CAC", mode="lines+markers",
        line=dict(color=METRIC_COLOR["CAC"], width=2),
        marker=dict(size=7, line=dict(width=1.5, color=SURFACE)),
        hovertemplate="<b>Blended CAC</b> · %{x}<br>AED %{y:,.0f}<extra></extra>",
    ))
    fig.update_yaxes(title_text="AED per converted customer",
                     title_font=dict(size=11, color=INK_MUTED), rangemode="tozero")
    return _style(fig)  # single series -> title carries identity, no legend


def chart_cpl_cpql_trend(spend: pd.DataFrame) -> go.Figure:
    agg = _cost_agg(spend)
    fig = go.Figure()
    for metric, label in [("CPL", "Cost per lead"), ("CPQL", "Cost per qualified lead")]:
        fig.add_trace(go.Scatter(
            x=agg["month"], y=agg[metric], name=label, mode="lines+markers",
            line=dict(color=METRIC_COLOR[metric], width=2),
            marker=dict(size=7, line=dict(width=1.5, color=SURFACE)),
            hovertemplate=f"<b>{label}</b> · %{{x}}<br>AED %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_yaxes(title_text="AED per lead", title_font=dict(size=11, color=INK_MUTED),
                     rangemode="tozero")
    return _style(fig)


# ---- 6B. ROAS by channel & vertical (grouped bar; color = vertical) -----

def chart_roas(roas: pd.DataFrame) -> go.Figure:
    summ = (roas.groupby(["channel", "vertical"], as_index=False)
            .agg(revenue=("revenue_aed", "sum"), spend=("allocated_spend_aed", "sum")))
    summ = summ[summ["vertical"].isin(VERTICALS)]
    summ["roas"] = summ.apply(lambda r: r["revenue"] / r["spend"] if r["spend"] else None, axis=1)
    channel_order = (summ.groupby("channel")["revenue"].sum()
                     .sort_values(ascending=False).index.tolist())
    fig = go.Figure()
    for vert in VERTICALS:
        if vert not in sel_verticals:
            continue
        sub = summ[summ["vertical"] == vert].set_index("channel").reindex(channel_order).reset_index()
        fig.add_trace(go.Bar(
            x=sub["channel"], y=sub["roas"], name=vert,
            marker=dict(color=VERTICAL_COLOR[vert], line=dict(width=2, color=SURFACE)),
            hovertemplate=f"<b>{vert}</b> · %{{x}}<br>ROAS %{{y:,.0f}}×<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_yaxes(title_text="ROAS (×)", title_font=dict(size=11, color=INK_MUTED))
    fig.update_xaxes(tickangle=-30)
    return _style(fig, height=420)


# ---- 6C. Horizontal conversion funnel (ordinal blue ramp) ---------------

def chart_funnel(funnel: pd.DataFrame) -> go.Figure:
    stages = {
        "Lead": funnel["leads"].sum(),
        "Qualified": funnel["qualified"].sum(),
        "Site Visit": funnel["site_visit"].sum(),
        "Won": funnel["won"].sum(),
    }
    labels = list(stages.keys())
    values = list(stages.values())
    fig = go.Figure(go.Funnel(
        y=labels, x=values, orientation="h",
        marker=dict(color=[FUNNEL_STAGE_COLOR[s] for s in labels],
                    line=dict(width=2, color=SURFACE)),
        textposition="inside", textinfo="value+percent initial",
        textfont=dict(color="#ffffff", size=13),
        connector=dict(line=dict(color=GRID, width=1)),
        hovertemplate="<b>%{y}</b><br>%{x:,} leads<br>%{percentInitial} of top<extra></extra>",
    ))
    return _style(fig, height=360)


# ---- 6D. Days-to-convert by channel (single hue, sorted, with spread) ---

def chart_velocity(funnel: pd.DataFrame) -> go.Figure:
    v = funnel.dropna(subset=["mean_days"]).sort_values("mean_days")
    fig = go.Figure(go.Bar(
        x=v["mean_days"], y=v["channel"], orientation="h",
        marker=dict(color=SINGLE_HUE, line=dict(width=2, color=SURFACE)),
        error_x=dict(type="data", symmetric=False,
                     array=(v["p75_days"] - v["mean_days"]).clip(lower=0),
                     arrayminus=(v["mean_days"] - v["p25_days"]).clip(lower=0),
                     color=INK_MUTED, thickness=1.2, width=4),
        hovertemplate="<b>%{y}</b><br>mean %{x:.0f} days"
                      "<br>IQR %{customdata[0]:.0f}–%{customdata[1]:.0f}<extra></extra>",
        customdata=v[["p25_days", "p75_days"]].values,
    ))
    fig.update_xaxes(title_text="Avg days to convert (bar) · IQR (whiskers)",
                     title_font=dict(size=11, color=INK_MUTED))
    return _style(fig, height=420)


# ---- layout: 2 x 2 chart grid -------------------------------------------

section("Spend efficiency trend")
sc_l, sc_r = st.columns(2, gap="large")
with sc_l:
    st.markdown("**Blended CAC** · cost per converted customer")
    st.plotly_chart(chart_cac_trend(f_spend), use_container_width=True,
                    config={"displayModeBar": False})
with sc_r:
    st.markdown("**CPL vs CPQL** · cost per lead and per qualified lead")
    st.plotly_chart(chart_cpl_cpql_trend(f_spend), use_container_width=True,
                    config={"displayModeBar": False})

col_l, col_r = st.columns(2, gap="large")
with col_l:
    section("ROAS by channel & vertical", "Allocated-spend ROAS; color encodes vertical.")
    st.plotly_chart(chart_roas(f_roas), use_container_width=True)
with col_r:
    section("Conversion funnel", "Lead → Qualified → Site Visit → Won across selected channels.")
    st.plotly_chart(chart_funnel(f_funnel), use_container_width=True)

section("Conversion velocity", "Average days-to-convert by channel, with inter-quartile spread.")
st.plotly_chart(chart_velocity(f_funnel), use_container_width=True)


# ==========================================================================
# 7. AUTONOMOUS AI-INSIGHT CONTAINER  --  Claude CoPilot over MCP
# ==========================================================================

def build_dashboard_state() -> dict:
    """Serialize the live, filtered dashboard into a compact JSON state that a
    Claude CoPilot can read via an MCP tool call. This is the exact payload the
    MCP server would receive as `dashboard_state`."""
    roas_agg = f_roas.groupby("channel", as_index=False).agg(
        rev=("revenue_aed", "sum"), spd=("allocated_spend_aed", "sum"))
    top_roas = (roas_agg.assign(v=_ratio(roas_agg["rev"], roas_agg["spd"]))
                .dropna(subset=["v"]).set_index("channel")["v"].sort_values(ascending=False))
    cac_agg = f_spend.groupby("channel", as_index=False).agg(
        spd=("actual_spend_aed", "sum"), conv=("crm_converted_customers", "sum"))
    cac_by_channel = (cac_agg.assign(v=_ratio(cac_agg["spd"], cac_agg["conv"]))
                      .dropna(subset=["v"]).set_index("channel")["v"].sort_values())
    return {
        "filters": {"verticals": sel_verticals, "months": [month_start, month_end],
                    "channels": sel_channels},
        "kpis": {"total_revenue_aed": float(total_revenue),
                 "total_ad_spend_aed": float(total_spend),
                 "blended_cac_aed": None if pd.isna(blended_cac) else float(blended_cac),
                 "overall_roas": None if pd.isna(overall_roas) else float(overall_roas)},
        "leaderboards": {
            "highest_roas_channels": top_roas.head(3).round(0).to_dict(),
            "lowest_cac_channels": cac_by_channel.head(3).round(0).to_dict(),
            "highest_cac_channels": cac_by_channel.tail(3).round(0).to_dict(),
        },
        "funnel_totals": {"leads": int(f_funnel["leads"].sum()),
                          "qualified": int(f_funnel["qualified"].sum()),
                          "site_visit": int(f_funnel["site_visit"].sum()),
                          "won": int(f_funnel["won"].sum())},
    }


def query_mcp_copilot(state: dict) -> str | None:
    """Integration point for the Claude CoPilot MCP server.

    In production this posts `state` to the MCP endpoint (configured via the
    PRYPCO_MCP_ENDPOINT env var), where a Claude-backed tool returns a board-
    ready summary. Kept behind a guard so the dashboard renders with a
    deterministic local fallback when the server is not wired up.
    """
    endpoint = os.environ.get("PRYPCO_MCP_ENDPOINT")
    if not endpoint:
        return None
    try:
        import requests  # local import: only needed when MCP is configured
        resp = requests.post(
            endpoint,
            json={"tool": "summarize_dashboard", "arguments": {"dashboard_state": state}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("summary")
    except Exception as exc:  # never let the CoPilot panel break the dashboard
        st.session_state["_mcp_error"] = str(exc)
        return None


def local_fallback_summary(state: dict) -> str:
    """Deterministic, data-grounded summary so the panel is useful offline."""
    k = state["kpis"]
    lb = state["leaderboards"]
    fn = state["funnel_totals"]
    l2q = 100 * fn["qualified"] / fn["leads"] if fn["leads"] else 0
    q2w = 100 * fn["won"] / fn["qualified"] if fn["qualified"] else 0
    best = next(iter(lb["highest_roas_channels"]), "—")
    cheapest = next(iter(lb["lowest_cac_channels"]), "—")
    priciest = list(lb["highest_cac_channels"])[-1] if lb["highest_cac_channels"] else "—"
    cac_txt = "n/a" if k["blended_cac_aed"] is None else f"AED {k['blended_cac_aed']:,.0f}"
    return (
        f"Across the selected window, PRYPCO booked **{_fmt_aed(k['total_revenue_aed'])}** in "
        f"attributed revenue on **{_fmt_aed(k['total_ad_spend_aed'])}** of ad spend, a blended "
        f"CAC of **{cac_txt}** per converted customer. The lead→qualified rate holds at "
        f"**{l2q:.0f}%** and qualified→won at **{q2w:.0f}%**, so the leak is concentrated below "
        f"the qualification stage, not at the top of the funnel. **{cheapest}** is the most "
        f"capital-efficient acquisition channel and **{best}** carries the strongest ROAS, while "
        f"**{priciest}** is the costliest per conversion and is the first candidate for budget "
        f"reallocation. Recommend shifting marginal spend toward the low-CAC, high-intent channels "
        f"and instrumenting the qualified→site-visit hand-off, where the largest drop-off sits."
    )


section("AI CoPilot — executive summary",
        "Claude reads the live dashboard state over MCP and drafts a board-ready readout.")

state = build_dashboard_state()

with st.container():
    st.markdown('<div class="copilot-shell">', unsafe_allow_html=True)
    top = st.columns([0.72, 0.28])
    with top[0]:
        st.markdown('<span class="copilot-badge">◆ Claude CoPilot · MCP</span>',
                    unsafe_allow_html=True)
    with top[1]:
        run = st.button("Generate summary", type="primary", use_container_width=True)

    if run or st.session_state.get("_copilot_ran"):
        st.session_state["_copilot_ran"] = True
        summary = query_mcp_copilot(state)
        source = "Claude via MCP server"
        if summary is None:
            summary = local_fallback_summary(state)
            source = "Local fallback (set PRYPCO_MCP_ENDPOINT to route through Claude/MCP)"
        st.markdown(f"<div style='margin-top:1rem; font-size:.98rem; line-height:1.6; "
                    f"color:var(--ink-1);'>{summary}</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="copilot-meta">Source: {source} &nbsp;·&nbsp; '
                    f'state hash {abs(hash(json.dumps(state, sort_keys=True))) % 10**8:08d}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top:1rem; color:var(--ink-2); font-size:.92rem;'>"
                    "Press <b>Generate summary</b> to have the CoPilot read the current filter "
                    "state and KPIs and draft an executive readout.</div>", unsafe_allow_html=True)

    with st.expander("Inspect the MCP payload (dashboard_state)"):
        st.json(state)
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# 8. DATA / METHOD FOOTNOTES
# ==========================================================================

with st.expander("Methodology & data-integrity notes"):
    st.markdown(
        """
- **Revenue** counts only `Completed` orders; `Pending`/`Cancelled`/`Refunded` are excluded.
- **Vertical-level ROAS** allocates each channel's monthly spend across verticals in proportion
  to that channel's completed-order revenue share (source spend is channel-level, not per vertical).
  Spend with no attributed revenue is bucketed as *Unallocated* rather than force-fitted.
- **Blended CAC** = actual spend ÷ converted customers (`Won` status or a `Completed` order).
- **Funnel & days-to-convert** are channel-grain, all-time; they respond to the channel filter
  but not to vertical/month, by design, because the staging grain does not carry those dimensions.
- Upstream cleaning (dedup of `customer_id`, clipped impossible `qualified_leads`, orphan-order
  quarantine, orders-before-signup flag) is handled in `data_pipeline.py`; see the data audit.
        """
    )
