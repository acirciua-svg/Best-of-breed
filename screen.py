#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# python screen.py --input data/fundamentals.csv --outdir outputs

# —— Parameters BoB (editable) ——
DM_MC_MIN = 25e9      # Developed Markets: market cap min USD
DM_ADV_MIN = 25e6     # Developed Markets: average dollar volume min USD
EM_MC_MIN = 10e9      # Emerging Markets: market cap min USD
EM_ADV_MIN = 10e6     # Emerging Markets: average dollar volume min USD

VALID_RATINGS = {"Buy", "Neutral"}

# —— Utility —— 

# —— Function to normalize data into booleans —— 
def _bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return bool(x)


# —— Function to check the liquidity of the company —— 

def _liquidity_ok(row):
    if row["market"] == "DM":
        return (row["mkt_cap_usd"] >= DM_MC_MIN) and (row["avg_dollar_vol_usd"] >= DM_ADV_MIN)
    return (row["mkt_cap_usd"] >= EM_MC_MIN) and (row["avg_dollar_vol_usd"] >= EM_ADV_MIN)

# —— Function to check the Investment opinion of the analysts —— 

def _opinion_ok(row):
    return str(row["analyst_rating"]).strip() in VALID_RATINGS

# —— Function to check if there are any missing columns  —— 

def _require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"ERRORE: missing columns in input CSV: {missing}")

# —— Function to calculate the median value by sector  —— 

def _sector_medians(df, cols):
    return df.groupby("sector")[cols].median().rename(columns=lambda c: f"sector_median_{c}")

# —— Function to convert possible strings into numbers ——
def _as_number(s):
    try:
        return float(s)
    except Exception:
        return np.nan
    
# —— Function to load the data from the CSV in input——

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize basic types
    df["is_financial"] = df["is_financial"].apply(_bool)
    df["market"] = df["market"].astype(str)
    df["sector"] = df["sector"].astype(str)
    # Ensure numeric values
    numeric_cols = [
        "mkt_cap_usd","avg_dollar_vol_usd",
        "eps_g_t","eps_g_t1","eps_g_t2",
        "roic_t","roic_t1",
        "netdebt_to_equity_t","netdebt_to_equity_t1",
        "current_ratio_t",
        "ebitda_margin_t","ebitda_margin_t1",
        "roa_t","roa_t1",
        "tier1_ratio_t","tier1_ratio_t1",
        "llp_to_npl_t","llp_to_npl_t1",
        "nim_t","nim_t1",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].apply(_as_number)
    return df

# —— Function to check nf/fin missing columns——

def validate_schema(df: pd.DataFrame):
    base = ["ticker","name","sector","is_financial","market","analyst_rating","mkt_cap_usd","avg_dollar_vol_usd","eps_g_t","eps_g_t1","eps_g_t2"]
    nf = ["roic_t","roic_t1","netdebt_to_equity_t","netdebt_to_equity_t1","current_ratio_t","ebitda_margin_t","ebitda_margin_t1"]
    fin = ["roa_t","roa_t1","tier1_ratio_t","tier1_ratio_t1","llp_to_npl_t","llp_to_npl_t1","nim_t","nim_t1"]
    _require_columns(df, base)
    # I don’t require all NF/FIN fields to be present, but I issue a warning
    missing_nf = [c for c in nf if c not in df.columns]
    missing_fin = [c for c in fin if c not in df.columns]
    if missing_nf:
        print(f"ATTENZIONE: columns Non-Financials missing: {missing_nf}", file=sys.stderr)
    if missing_fin:
        print(f"ATTENZIONE: columns Financials missing: {missing_fin}", file=sys.stderr)

# —— Applying conditions to Non-Financials to be considered in the portfolio——

def apply_non_financial_screens(df: pd.DataFrame) -> pd.DataFrame:
    nf = df[~df["is_financial"]].copy()
    if nf.empty:
        return nf.assign(passes_all=False, fail_reason="no_non_financials")
    cols = ["eps_g_t","eps_g_t1","roic_t","roic_t1","netdebt_to_equity_t","netdebt_to_equity_t1","current_ratio_t","ebitda_margin_t","ebitda_margin_t1"]
    med = _sector_medians(nf, [c for c in cols if c in nf.columns])
    nf = nf.join(med, on="sector")

    reasons = []

    cond_liq = nf.apply(_liquidity_ok, axis=1)
    reasons.append((~cond_liq, "liquidity"))

    cond_op = nf.apply(_opinion_ok, axis=1)
    reasons.append((~cond_op, "opinion"))

    # EPS growth: (> median t & t+1) OR (t+2 > t+1)
    cond_eps = ((nf["eps_g_t"] > nf.get("sector_median_eps_g_t", np.nan)) &
                (nf["eps_g_t1"] > nf.get("sector_median_eps_g_t1", np.nan))) | \
               (nf["eps_g_t2"] > nf["eps_g_t1"])
    reasons.append((~cond_eps, "eps_growth"))

    # ROIC: (> median t & t+1) OR (roic_t1 > roic_t)
    cond_roic = ((nf["roic_t"] > nf.get("sector_median_roic_t", np.nan)) &
                 (nf["roic_t1"] > nf.get("sector_median_roic_t1", np.nan))) | \
                (nf["roic_t1"] > nf["roic_t"])
    reasons.append((~cond_roic, "roic"))

    # Net Debt/Equity: (< median t & t+1) OR (decline)
    cond_nde = ((nf["netdebt_to_equity_t"] < nf.get("sector_median_netdebt_to_equity_t", np.nan)) &
                (nf["netdebt_to_equity_t1"] < nf.get("sector_median_netdebt_to_equity_t1", np.nan))) | \
               (nf["netdebt_to_equity_t1"] < nf["netdebt_to_equity_t"])
    reasons.append((~cond_nde, "net_debt_to_equity"))

    # Current ratio: > median (t)
    cond_curr = nf["current_ratio_t"] > nf.get("sector_median_current_ratio_t", np.nan)
    reasons.append((~cond_curr, "current_ratio"))

    # EBITDA margin: (> median t & t+1) OR (improving)
    cond_marg = ((nf["ebitda_margin_t"] > nf.get("sector_median_ebitda_margin_t", np.nan)) &
                 (nf["ebitda_margin_t1"] > nf.get("sector_median_ebitda_margin_t1", np.nan))) | \
                (nf["ebitda_margin_t1"] > nf["ebitda_margin_t"])
    reasons.append((~cond_marg, "ebitda_margin"))

    # Final result 
    nf["passes_all"] = cond_liq & cond_op & cond_eps & cond_roic & cond_nde & cond_curr & cond_marg
    nf["fail_reason"] = ""
    for mask, label in reasons:
        nf.loc[mask & ~nf["passes_all"], "fail_reason"] = nf["fail_reason"].where(nf["fail_reason"]=="", nf["fail_reason"] + "|") + label
    return nf

# —— Applying conditions for Financials to be considered in the portfolio——

def apply_financial_screens(df: pd.DataFrame) -> pd.DataFrame:
    f = df[df["is_financial"]].copy()
    if f.empty:
        return f.assign(passes_all=False, fail_reason="no_financials")
    cols = ["eps_g_t","eps_g_t1","roa_t","roa_t1","tier1_ratio_t","tier1_ratio_t1","llp_to_npl_t","llp_to_npl_t1","nim_t","nim_t1"]
    med = _sector_medians(f, [c for c in cols if c in f.columns])
    f = f.join(med, on="sector")

    reasons = []

    cond_liq = f.apply(_liquidity_ok, axis=1)
    reasons.append((~cond_liq, "liquidity"))

    cond_op = f.apply(_opinion_ok, axis=1)
    reasons.append((~cond_op, "opinion"))

    # EPS growth: (> median t & t+1) OR (t+1 > t)
    cond_eps = ((f["eps_g_t"] > f.get("sector_median_eps_g_t", np.nan)) &
                (f["eps_g_t1"] > f.get("sector_median_eps_g_t1", np.nan))) | \
               (f["eps_g_t1"] > f["eps_g_t"])
    reasons.append((~cond_eps, "eps_growth"))

    # ROA: (> median t & t+1) OR (roa_t1 > roa_t)
    cond_roa = ((f["roa_t"] > f.get("sector_median_roa_t", np.nan)) &
                (f["roa_t1"] > f.get("sector_median_roa_t1", np.nan))) | \
               (f["roa_t1"] > f["roa_t"])
    reasons.append((~cond_roa, "roa"))

    # Tier1: (> median t & t+1) OR (growth)
    cond_tier1 = ((f["tier1_ratio_t"] > f.get("sector_median_tier1_ratio_t", np.nan)) &
                  (f["tier1_ratio_t1"] > f.get("sector_median_tier1_ratio_t1", np.nan))) | \
                 (f["tier1_ratio_t1"] > f["tier1_ratio_t"])
    reasons.append((~cond_tier1, "tier1_ratio"))

    # LLP/NPL: (> median t & t+1) OR (growth)
    cond_prov = ((f["llp_to_npl_t"] > f.get("sector_median_llp_to_npl_t", np.nan)) &
                 (f["llp_to_npl_t1"] > f.get("sector_median_llp_to_npl_t1", np.nan))) | \
                (f["llp_to_npl_t1"] > f["llp_to_npl_t"])
    reasons.append((~cond_prov, "llp_to_npl"))

    # NIM: (> median t & t+1) OR (growth)
    cond_nim = ((f["nim_t"] > f.get("sector_median_nim_t", np.nan)) &
                (f["nim_t1"] > f.get("sector_median_nim_t1", np.nan))) | \
               (f["nim_t1"] > f["nim_t"])
    reasons.append((~cond_nim, "nim"))

    f["passes_all"] = cond_liq & cond_op & cond_eps & cond_roa & cond_tier1 & cond_prov & cond_nim
    f["fail_reason"] = ""
    for mask, label in reasons:
        f.loc[mask & ~f["passes_all"], "fail_reason"] = f["fail_reason"].where(f["fail_reason"]=="", f["fail_reason"] + "|") + label
    return f

# —— Function to calculate weights based on market cap——

def cap_weight(eligible: pd.DataFrame) -> pd.DataFrame:
    if eligible.empty:
        return eligible.assign(weight=[])
    w = eligible.copy()
    total = w["mkt_cap_usd"].sum()
    w["weight"] = w["mkt_cap_usd"] / total
    return w.sort_values("weight", ascending=False)

# —— Main screening function——

def run_screen(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nf = apply_non_financial_screens(df)
    fin = apply_financial_screens(df)
    combined = pd.concat([nf, fin], ignore_index=True)
    eligible = combined[combined["passes_all"]].copy()
    weights = cap_weight(eligible[["ticker","name","sector","mkt_cap_usd","passes_all"]])
    return combined, eligible, weights

# —— Main function ——

def main():
    ap = argparse.ArgumentParser(description="Best of Breed-style screening")
    ap.add_argument("--input", required=True, help="Path CSV fundamentals in input")
    ap.add_argument("--outdir", default="outputs", help="Folder output (default: outputs)")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(in_path)
    validate_schema(df)

    combined, eligible, weights = run_screen(df)

    combined.to_csv(outdir / "diagnostics.csv", index=False)
    eligible.to_csv(outdir / "constituents.csv", index=False)
    weights.to_csv(outdir / "weights_market_cap.csv", index=False)

    print(f"\nTotal stocks: {len(df)}")
    print(f"Passed screening: {len(eligible)}")
    print(f"Files created: {outdir.resolve()}")
    print(" - diagnostics.csv  (reasons for passing/failing)")
    print(" - constituents.csv (list stocks)")
    print(" - weights_market_cap.csv (weights by market cap)")

if __name__ == "__main__":
    main()
