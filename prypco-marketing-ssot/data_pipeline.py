"""
PRYPCO Blocks -- SSOT Data Processing Pipeline
Loads customers.csv, orders.csv, marketing_spend.csv, cleans them, reconciles
relational integrity between customers and orders, and exports four staging
tables for the dashboard layer:

  1. staging_acquisition_conversion   -- signups & lead-to-qualified rate by channel/month/vertical
  2. staging_spend_efficiency         -- CAC, CPL, CPQL, budget-vs-actual by channel/month
  3. staging_funnel_velocity          -- Lead->Qualified->SiteVisit->Won funnel + days-to-convert by channel
  4. staging_vertical_roas            -- ROAS by channel and vertical

Run: python3 data_pipeline.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DATA_DIR = Path("/Users/falahkhanji/Downloads")
PIPELINE_DIR = DATA_DIR / "prypco_pipeline"
STAGING_DIR = PIPELINE_DIR / "staging"
QUARANTINE_DIR = PIPELINE_DIR / "quarantine"
CLEAN_DIR = PIPELINE_DIR / "clean"

CUSTOMERS_PATH = DATA_DIR / "customers - customers.csv.csv"
ORDERS_PATH = DATA_DIR / "orders - orders.csv.csv"
SPEND_PATH = DATA_DIR / "marketing_spend - marketing_spend.csv.csv"

VERTICALS = ["Primary Sales", "Secondary Sales", "Rental", "Mortgage", "Property Management"]

# Ordinal funnel stages, built from lead_status. lead_status is a single
# current-state snapshot with no stage-transition timestamps, so "reached
# stage X" is inferred as "current status is X or any status further down
# the funnel." Lost/Dormant customers keep whatever stage they last reached.
FUNNEL_ORDER = [
    "New Lead",
    "Contacted",
    "Qualified",
    "Site Visit Scheduled",
    "Site Visit Done",
    "Negotiation",
    "Won",
]
TERMINAL_DROPOUT_STATUSES = {"Lost", "Dormant"}
QUALIFIED_OR_BEYOND = {"Qualified", "Site Visit Scheduled", "Site Visit Done", "Negotiation", "Won"}
SITE_VISIT_OR_BEYOND = {"Site Visit Done", "Negotiation", "Won"}
WON_STATUS = {"Won"}

REVENUE_ORDER_STATUSES = {"Completed"}  # only Completed orders count as realized revenue
CHANNEL_TEXT_COLUMNS_CUSTOMERS = [
    "first_touch_channel", "last_touch_channel", "utm_source", "utm_medium",
    "utm_campaign", "device", "platform", "city", "nationality", "lead_status",
    "vertical_interest",
]
CHANNEL_TEXT_COLUMNS_ORDERS = [
    "vertical", "product", "order_status", "payment_method",
    "attributed_channel", "attributed_campaign",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("prypco_pipeline")


@dataclass
class CleaningReport:
    """Tallies of what the cleaning step touched, for the run summary."""
    name: str
    rows_in: int = 0
    rows_out: int = 0
    duplicates_dropped: int = 0
    quarantined: int = 0
    notes: list | None = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def log_summary(self):
        log.info(
            "%s: %d rows in -> %d rows out | duplicates_dropped=%d quarantined=%d",
            self.name, self.rows_in, self.rows_out, self.duplicates_dropped, self.quarantined,
        )
        for note in self.notes:
            log.info("  - %s", note)


def _ensure_dirs() -> None:
    for d in (STAGING_DIR, QUARANTINE_DIR, CLEAN_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _normalize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected input file not found: {path}")
    df = pd.read_csv(path)
    log.info("Loaded %s: %d rows, %d columns", path.name, len(df), len(df.columns))
    return df


# --------------------------------------------------------------------------
# Cleaning: customers.csv
# --------------------------------------------------------------------------

def clean_customers(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(name="customers", rows_in=len(raw))
    df = raw.copy()

    df = _normalize_text_columns(df, CHANNEL_TEXT_COLUMNS_CUSTOMERS)

    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["signup_datetime"] = pd.to_datetime(df["signup_datetime"], errors="coerce")
    unparseable = df[df["signup_date"].isna() | df["signup_datetime"].isna()]
    if len(unparseable):
        report.notes.append(f"{len(unparseable)} rows had unparseable signup dates and were dropped")
        unparseable.to_csv(QUARANTINE_DIR / "customers_unparseable_dates.csv", index=False)
        df = df.drop(index=unparseable.index)

    # signup_datetime carries finer grain; trust it and regenerate signup_date from it
    df["signup_date"] = df["signup_datetime"].dt.normalize()
    df["signup_month"] = df["signup_datetime"].dt.to_period("M").astype(str)

    numeric_cols = [
        "sessions_count", "pages_viewed", "time_on_site_minutes", "properties_viewed",
        "enquiries_submitted", "site_visits_booked", "app_installed",
        "whatsapp_opted_in", "email_opted_in",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # engagement counters: a missing value means "no tracking event recorded,"
    # which is functionally zero for aggregation purposes, not an unknown
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df["mmp_data_missing_flag"] = (df["app_installed"] == 1) & (
        df["mmp_source"].isna() | df["mmp_network"].isna() | df["mmp_click_id"].isna()
    )

    # duplicate customer_id: keep the most recent signup_datetime per ID and
    # quarantine the rest for manual review rather than silently discarding
    dup_mask = df.duplicated(subset="customer_id", keep=False)
    if dup_mask.any():
        dupes = df[dup_mask].sort_values(["customer_id", "signup_datetime"])
        dupes.to_csv(QUARANTINE_DIR / "customers_duplicate_ids.csv", index=False)
        df = df.sort_values("signup_datetime").drop_duplicates(subset="customer_id", keep="last")
        report.duplicates_dropped = int(dup_mask.sum()) - df["customer_id"].duplicated().sum()
        report.notes.append(
            f"{dup_mask.sum()} rows shared a duplicate customer_id; kept latest signup_datetime "
            f"per ID, quarantined all {dup_mask.sum()} original rows for review"
        )

    df["vertical_interest"] = df["vertical_interest"].where(
        df["vertical_interest"].isin(VERTICALS), other=pd.NA
    )

    report.rows_out = len(df)
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Cleaning: orders.csv
# --------------------------------------------------------------------------

def clean_orders(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(name="orders", rows_in=len(raw))
    df = raw.copy()

    df = _normalize_text_columns(df, CHANNEL_TEXT_COLUMNS_ORDERS)

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")
    unparseable = df[df["order_date"].isna() | df["order_datetime"].isna()]
    if len(unparseable):
        report.notes.append(f"{len(unparseable)} rows had unparseable order dates and were dropped")
        unparseable.to_csv(QUARANTINE_DIR / "orders_unparseable_dates.csv", index=False)
        df = df.drop(index=unparseable.index)

    df["order_month"] = df["order_datetime"].dt.to_period("M").astype(str)

    numeric_cols = [
        "transaction_value_aed", "commission_aed", "ad_spend_attributed_aed",
        "days_to_convert", "touchpoints_before_conversion",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_financials = df[(df["transaction_value_aed"].isna()) | (df["transaction_value_aed"] <= 0)]
    if len(bad_financials):
        bad_financials.to_csv(QUARANTINE_DIR / "orders_bad_transaction_value.csv", index=False)
        report.quarantined += len(bad_financials)
        report.notes.append(f"{len(bad_financials)} rows had null/zero/negative transaction_value_aed and were dropped")
        df = df.drop(index=bad_financials.index)

    dup_mask = df.duplicated(subset="order_id", keep=False)
    if dup_mask.any():
        dupes = df[dup_mask]
        dupes.to_csv(QUARANTINE_DIR / "orders_duplicate_ids.csv", index=False)
        df = df.drop_duplicates(subset="order_id", keep="first")
        report.duplicates_dropped = int(dup_mask.sum())
        report.notes.append(f"{dup_mask.sum()} duplicate order_id rows quarantined, kept first occurrence")

    df["vertical"] = df["vertical"].where(df["vertical"].isin(VERTICALS), other=pd.NA)
    unknown_vertical = df["vertical"].isna().sum()
    if unknown_vertical:
        report.notes.append(f"{unknown_vertical} rows had a vertical outside the 5 known verticals")

    report.rows_out = len(df)
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Cleaning: marketing_spend.csv
# --------------------------------------------------------------------------

def clean_marketing_spend(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(name="marketing_spend", rows_in=len(raw))
    df = raw.copy()

    df["channel"] = df["channel"].astype("string").str.strip()
    df["month"] = pd.PeriodIndex(df["month"], freq="M").astype(str)

    numeric_cols = [
        "budget_allocated_aed", "actual_spend_aed", "impressions", "clicks",
        "leads_generated", "qualified_leads",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dup_mask = df.duplicated(subset=["month", "channel"], keep=False)
    if dup_mask.any():
        df[dup_mask].to_csv(QUARANTINE_DIR / "spend_duplicate_month_channel.csv", index=False)
        df = df.drop_duplicates(subset=["month", "channel"], keep="first")
        report.duplicates_dropped = int(dup_mask.sum())
        report.notes.append(f"{dup_mask.sum()} duplicate month+channel rows quarantined")

    # a lead must exist before it can be qualified: qualified_leads can never
    # exceed leads_generated. Clip to the valid ceiling and quarantine the
    # original rows for the source-system team to investigate (likely a
    # swapped-column export bug or double-counted qualified leads)
    impossible = df[df["qualified_leads"] > df["leads_generated"]]
    if len(impossible):
        impossible.to_csv(QUARANTINE_DIR / "spend_qualified_exceeds_leads.csv", index=False)
        report.quarantined += len(impossible)
        report.notes.append(
            f"{len(impossible)} rows had qualified_leads > leads_generated; "
            f"clipped qualified_leads to leads_generated, originals quarantined"
        )
        df.loc[df["qualified_leads"] > df["leads_generated"], "qualified_leads"] = df["leads_generated"]

    df["ctr_pct"] = _safe_divide(df["clicks"], df["impressions"]) * 100
    df["implausible_ctr_flag"] = (df["ctr_pct"] > 15) | (df["clicks"] > df["impressions"])

    df["budget_variance_aed"] = df["actual_spend_aed"] - df["budget_allocated_aed"]
    df["budget_variance_pct"] = _safe_divide(df["budget_variance_aed"], df["budget_allocated_aed"]) * 100

    report.rows_out = len(df)
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Relational integrity: reconcile customers <-> orders
# --------------------------------------------------------------------------

def reconcile_customers_orders(
    customers: pd.DataFrame, orders: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    report = CleaningReport(name="reconciliation", rows_in=len(orders))

    known_ids = set(customers["customer_id"])
    orphan_mask = ~orders["customer_id"].isin(known_ids)
    if orphan_mask.any():
        orders[orphan_mask].to_csv(QUARANTINE_DIR / "orders_orphan_customer_id.csv", index=False)
        report.quarantined = int(orphan_mask.sum())
        report.notes.append(
            f"{orphan_mask.sum()} orders reference a customer_id absent from customers.csv; "
            f"excluded from revenue/attribution joins, quarantined for backfill investigation"
        )
    orders_reconciled = orders[~orphan_mask].copy()

    signup_lookup = customers.set_index("customer_id")["signup_date"]
    orders_reconciled["customer_signup_date"] = orders_reconciled["customer_id"].map(signup_lookup)
    orders_reconciled["order_before_signup_flag"] = (
        orders_reconciled["order_date"] < orders_reconciled["customer_signup_date"]
    )
    flagged = int(orders_reconciled["order_before_signup_flag"].sum())
    if flagged:
        report.notes.append(
            f"{flagged} orders are dated before the customer's signup_date; kept in the "
            f"dataset (revenue is real) but flagged via order_before_signup_flag for review"
        )

    report.rows_out = len(orders_reconciled)
    return customers, orders_reconciled, report


# --------------------------------------------------------------------------
# Staging 1: Customer Acquisition & Conversion
# --------------------------------------------------------------------------

def build_acquisition_conversion_staging(customers: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = customers.copy()
    df["reached_qualified"] = df["lead_status"].isin(QUALIFIED_OR_BEYOND)

    detail = (
        df.groupby(["first_touch_channel", "signup_month", "vertical_interest"], dropna=False)
        .agg(leads=("customer_id", "count"), qualified=("reached_qualified", "sum"))
        .reset_index()
        .rename(columns={"first_touch_channel": "channel", "vertical_interest": "vertical"})
    )
    detail["lead_to_qualified_rate_pct"] = _safe_divide(detail["qualified"], detail["leads"]) * 100

    channel_summary = (
        df.groupby("first_touch_channel", dropna=False)
        .agg(leads=("customer_id", "count"), qualified=("reached_qualified", "sum"))
        .reset_index()
        .rename(columns={"first_touch_channel": "channel"})
    )
    channel_summary["lead_to_qualified_rate_pct"] = _safe_divide(
        channel_summary["qualified"], channel_summary["leads"]
    ) * 100
    channel_summary = channel_summary.sort_values("leads", ascending=False)

    return {
        "staging_acquisition_conversion": detail,
        "staging_acquisition_conversion_channel_summary": channel_summary,
    }


# --------------------------------------------------------------------------
# Staging 2: Spend Efficiency (CAC, CPL, CPQL, budget variance)
# --------------------------------------------------------------------------

def build_spend_efficiency_staging(
    customers: pd.DataFrame, orders: pd.DataFrame, spend: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    completed_customer_ids = set(
        orders.loc[orders["order_status"].isin(REVENUE_ORDER_STATUSES), "customer_id"]
    )
    cust = customers.copy()
    cust["converted"] = cust["lead_status"].isin(WON_STATUS) | cust["customer_id"].isin(completed_customer_ids)

    acquisition = (
        cust.groupby(["first_touch_channel", "signup_month"], dropna=False)
        .agg(crm_leads=("customer_id", "count"), crm_converted_customers=("converted", "sum"))
        .reset_index()
        .rename(columns={"first_touch_channel": "channel", "signup_month": "month"})
    )

    detail = spend.merge(acquisition, on=["channel", "month"], how="left")
    detail[["crm_leads", "crm_converted_customers"]] = detail[["crm_leads", "crm_converted_customers"]].fillna(0)

    # CAC uses converted (paying) customers, not raw lead volume -- that is
    # what "customer acquisition cost" means financially. CPL/CPQL below use
    # marketing_spend's own platform-reported lead counts, per the assignment
    # spec; crm_* columns are kept alongside as a cross-check, since the CRM's
    # recorded lead volume and the ad platform's reported lead volume can
    # differ (a real SSOT reconciliation gap worth watching on the dashboard)
    detail["cac_aed"] = _safe_divide(detail["actual_spend_aed"], detail["crm_converted_customers"])
    detail["cpl_aed"] = _safe_divide(detail["actual_spend_aed"], detail["leads_generated"])
    detail["cpql_aed"] = _safe_divide(detail["actual_spend_aed"], detail["qualified_leads"])
    detail["cpl_aed_crm_cross_check"] = _safe_divide(detail["actual_spend_aed"], detail["crm_leads"])

    detail = detail[[
        "channel", "month", "budget_allocated_aed", "actual_spend_aed",
        "budget_variance_aed", "budget_variance_pct", "impressions", "clicks",
        "leads_generated", "qualified_leads", "crm_leads", "crm_converted_customers",
        "cac_aed", "cpl_aed", "cpql_aed", "cpl_aed_crm_cross_check",
    ]].sort_values(["channel", "month"])

    channel_summary = (
        detail.groupby("channel", dropna=False)
        .agg(
            total_budget_aed=("budget_allocated_aed", "sum"),
            total_actual_spend_aed=("actual_spend_aed", "sum"),
            total_leads_generated=("leads_generated", "sum"),
            total_qualified_leads=("qualified_leads", "sum"),
            total_crm_converted_customers=("crm_converted_customers", "sum"),
        )
        .reset_index()
    )
    channel_summary["cac_aed"] = _safe_divide(
        channel_summary["total_actual_spend_aed"], channel_summary["total_crm_converted_customers"]
    )
    channel_summary["cpl_aed"] = _safe_divide(
        channel_summary["total_actual_spend_aed"], channel_summary["total_leads_generated"]
    )
    channel_summary["cpql_aed"] = _safe_divide(
        channel_summary["total_actual_spend_aed"], channel_summary["total_qualified_leads"]
    )
    channel_summary["budget_variance_pct"] = _safe_divide(
        channel_summary["total_actual_spend_aed"] - channel_summary["total_budget_aed"],
        channel_summary["total_budget_aed"],
    ) * 100
    channel_summary = channel_summary.sort_values("total_actual_spend_aed", ascending=False)

    return {
        "staging_spend_efficiency": detail,
        "staging_spend_efficiency_channel_summary": channel_summary,
    }


# --------------------------------------------------------------------------
# Staging 3: Funnel + Days-to-Convert Velocity
# --------------------------------------------------------------------------

def build_funnel_velocity_staging(customers: pd.DataFrame, orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = customers.copy()
    df["reached_qualified"] = df["lead_status"].isin(QUALIFIED_OR_BEYOND)
    df["reached_site_visit"] = df["lead_status"].isin(SITE_VISIT_OR_BEYOND)
    df["reached_won"] = df["lead_status"].isin(WON_STATUS)

    funnel = (
        df.groupby("first_touch_channel", dropna=False)
        .agg(
            leads=("customer_id", "count"),
            qualified=("reached_qualified", "sum"),
            site_visit=("reached_site_visit", "sum"),
            won=("reached_won", "sum"),
        )
        .reset_index()
        .rename(columns={"first_touch_channel": "channel"})
    )
    funnel["lead_to_qualified_pct"] = _safe_divide(funnel["qualified"], funnel["leads"]) * 100
    funnel["qualified_to_site_visit_pct"] = _safe_divide(funnel["site_visit"], funnel["qualified"]) * 100
    funnel["site_visit_to_won_pct"] = _safe_divide(funnel["won"], funnel["site_visit"]) * 100
    funnel["overall_lead_to_won_pct"] = _safe_divide(funnel["won"], funnel["leads"]) * 100
    funnel = funnel.sort_values("leads", ascending=False)

    # days-to-convert velocity uses orders.attributed_channel, since that is
    # the order-level channel tag actually driving the conversion event
    # (it agrees with the customer's last-touch channel ~94% of the time;
    # see the data audit for the residual mismatch)
    valid_days = orders.dropna(subset=["attributed_channel", "days_to_convert"])
    velocity = (
        valid_days.groupby("attributed_channel")["days_to_convert"]
        .agg(
            orders_count="count",
            mean_days="mean",
            median_days="median",
            std_days="std",
            p25_days=lambda s: s.quantile(0.25),
            p75_days=lambda s: s.quantile(0.75),
            p90_days=lambda s: s.quantile(0.90),
            min_days="min",
            max_days="max",
        )
        .reset_index()
        .rename(columns={"attributed_channel": "channel"})
        .sort_values("mean_days")
    )

    return {
        "staging_funnel_by_channel": funnel,
        "staging_days_to_convert_by_channel": velocity,
    }


# --------------------------------------------------------------------------
# Staging 4: Vertical-Specific ROAS
# --------------------------------------------------------------------------

def build_vertical_roas_staging(orders: pd.DataFrame, spend: pd.DataFrame) -> dict[str, pd.DataFrame]:
    revenue_orders = orders[orders["order_status"].isin(REVENUE_ORDER_STATUSES)].copy()

    revenue_by_channel_month_vertical = (
        revenue_orders.groupby(["attributed_channel", "order_month", "vertical"], dropna=False)["transaction_value_aed"]
        .sum()
        .reset_index()
        .rename(columns={"attributed_channel": "channel", "order_month": "month", "transaction_value_aed": "revenue_aed"})
    )

    # marketing_spend.csv has no vertical breakdown, so a channel's monthly
    # spend is allocated across verticals in proportion to that channel's
    # Completed-order revenue share across verticals that same month. This is
    # a documented estimate, not raw data -- spend for a channel/month with
    # zero attributed revenue is kept as vertical="Unallocated" rather than
    # silently dropped or guessed
    revenue_share = revenue_by_channel_month_vertical.copy()
    channel_month_totals = revenue_share.groupby(["channel", "month"])["revenue_aed"].transform("sum")
    revenue_share["revenue_share_pct"] = _safe_divide(revenue_share["revenue_aed"], channel_month_totals)

    spend_slim = spend[["channel", "month", "actual_spend_aed"]]
    allocated = revenue_share.merge(spend_slim, on=["channel", "month"], how="outer")
    allocated["revenue_aed"] = allocated["revenue_aed"].fillna(0)
    allocated["vertical"] = allocated["vertical"].fillna("Unallocated")
    fallback_share = pd.Series(
        np.where(allocated["vertical"] == "Unallocated", 1.0, 0.0), index=allocated.index
    )
    allocated["revenue_share_pct"] = allocated["revenue_share_pct"].fillna(fallback_share)
    allocated["actual_spend_aed"] = allocated["actual_spend_aed"].fillna(0)
    allocated["allocated_spend_aed"] = allocated["actual_spend_aed"] * allocated["revenue_share_pct"]
    allocated["roas"] = _safe_divide(allocated["revenue_aed"], allocated["allocated_spend_aed"])

    detail = allocated[[
        "channel", "month", "vertical", "revenue_aed", "revenue_share_pct",
        "allocated_spend_aed", "roas",
    ]].sort_values(["channel", "month", "vertical"])

    summary = (
        detail.groupby(["channel", "vertical"], dropna=False)
        .agg(total_revenue_aed=("revenue_aed", "sum"), total_allocated_spend_aed=("allocated_spend_aed", "sum"))
        .reset_index()
    )
    summary["roas"] = _safe_divide(summary["total_revenue_aed"], summary["total_allocated_spend_aed"])
    summary = summary.sort_values(["channel", "total_revenue_aed"], ascending=[True, False])

    return {
        "staging_vertical_roas": detail,
        "staging_vertical_roas_summary": summary,
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def export_frames(frames: dict[str, pd.DataFrame], target_dir: Path) -> None:
    for name, frame in frames.items():
        path = target_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        log.info("Wrote %s (%d rows)", path.relative_to(PIPELINE_DIR), len(frame))


def main() -> None:
    _ensure_dirs()

    customers_raw = load_csv(CUSTOMERS_PATH)
    orders_raw = load_csv(ORDERS_PATH)
    spend_raw = load_csv(SPEND_PATH)

    customers, customers_report = clean_customers(customers_raw)
    orders, orders_report = clean_orders(orders_raw)
    spend, spend_report = clean_marketing_spend(spend_raw)
    for r in (customers_report, orders_report, spend_report):
        r.log_summary()

    customers, orders, reconciliation_report = reconcile_customers_orders(customers, orders)
    reconciliation_report.log_summary()

    export_frames({"customers_clean": customers, "orders_clean": orders, "marketing_spend_clean": spend}, CLEAN_DIR)

    staging_1 = build_acquisition_conversion_staging(customers)
    staging_2 = build_spend_efficiency_staging(customers, orders, spend)
    staging_3 = build_funnel_velocity_staging(customers, orders)
    staging_4 = build_vertical_roas_staging(orders, spend)

    for staging in (staging_1, staging_2, staging_3, staging_4):
        export_frames(staging, STAGING_DIR)

    log.info("Pipeline complete. Staging tables written to %s", STAGING_DIR)


if __name__ == "__main__":
    main()
