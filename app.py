import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

try:
    from tigramite.data_processing import DataFrame as TigramiteDataFrame
    from tigramite.independence_tests.parcorr import ParCorr
    try:
        from tigramite.independence_tests.gpdc import GPDC
        GPDC_AVAILABLE = True
    except ImportError:
        GPDC = None
        GPDC_AVAILABLE = False
    from tigramite.pcmci import PCMCI
    TIGRAMITE_AVAILABLE = True
except ImportError:
    TIGRAMITE_AVAILABLE = False
    GPDC_AVAILABLE = False
    TigramiteDataFrame = None
    ParCorr = None
    GPDC = None
    PCMCI = None

try:
    from streamlit_echarts import st_echarts
    ECHARTS_AVAILABLE = True
except ImportError:
    ECHARTS_AVAILABLE = False

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Causal Macro Dashboard", layout="wide")


@dataclass(frozen=True)
class SeriesSpec:
    file_name: str
    column: str
    label: str
    source_freq: str  # 'monthly' or 'daily'
    monthly_rule: str  # 'last' or 'mean'
    default_transform: str
    enabled_by_default: bool = True


SERIES_SPECS: List[SeriesSpec] = [
    SeriesSpec("macrotonghop.csv", "CPI_yoy", "CPI YoY", "monthly", "last", "level"),
    SeriesSpec("fdioverallrawdata.csv", "FDI_giai_ngan", "FDI Disbursed", "monthly", "last", "log_diff1"),
    SeriesSpec("khachqtrawdata.csv", "khachqte_all", "International Tourists", "monthly", "last", "log_diff1"),
    SeriesSpec("pmirawdata.csv", "pmi", "PMI", "monthly", "last", "diff1"),
    SeriesSpec("iipthangrawdata.csv", "IIP_yoy_revised", "IIP YoY (Revised)", "monthly", "last", "level"),
    SeriesSpec("dailycurrencyrawdata.csv", "VND=", "USD/VND", "daily", "last", "log_diff1"),
    SeriesSpec("dailygovbondzerocurve.csv", "VN10YT=HN", "VN 10Y Yield", "daily", "last", "diff1"),
    SeriesSpec("dailylslnhvnivnd.csv", "VNIVNDSWD=", "Interbank Rate", "daily", "mean", "diff1"),
    SeriesSpec("dailysavingratebig4.csv", "VND12MAV=", "12M Deposit Rate", "daily", "last", "diff1"),
]

TRANSFORM_OPTIONS = ["level", "diff1", "log", "log_diff1", "pct_change"]


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


@st.cache_data(show_spinner=False)
def load_monthly_series(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    loaded = {}
    meta_rows = []

    for spec in SERIES_SPECS:
        path = os.path.join(data_dir, spec.file_name)
        raw = safe_read_csv(path)
        if raw is None:
            meta_rows.append({
                "Series": spec.label,
                "Status": "Missing file or invalid date column",
                "Start": None,
                "End": None,
                "Observations": 0,
            })
            continue
        if spec.column not in raw.columns:
            meta_rows.append({
                "Series": spec.label,
                "Status": f"Missing column: {spec.column}",
                "Start": None,
                "End": None,
                "Observations": 0,
            })
            continue

        df = raw[["date", spec.column]].copy()
        df[spec.column] = pd.to_numeric(df[spec.column], errors="coerce")
        df = df.dropna(subset=[spec.column]).set_index("date").sort_index()

        if spec.source_freq == "daily":
            if spec.monthly_rule == "mean":
                df = df.resample("ME").mean()
            else:
                df = df.resample("ME").last()
        else:
            df = df.resample("ME").last()

        df = df.rename(columns={spec.column: spec.label})
        loaded[spec.label] = df

        non_na = df[spec.label].dropna()
        meta_rows.append({
            "Series": spec.label,
            "Status": "Loaded",
            "Start": non_na.index.min().date() if not non_na.empty else None,
            "End": non_na.index.max().date() if not non_na.empty else None,
            "Observations": int(non_na.shape[0]),
        })

    meta = pd.DataFrame(meta_rows)
    monthly = pd.concat(loaded.values(), axis=1).sort_index() if loaded else pd.DataFrame()
    return monthly, meta


def apply_transform(series: pd.Series, transform: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()
    if transform == "level":
        return s
    if transform == "diff1":
        return s.diff()
    if transform == "log":
        s = s.where(s > 0)
        return np.log(s)
    if transform == "log_diff1":
        s = s.where(s > 0)
        return np.log(s).diff()
    if transform == "pct_change":
        return s.pct_change()
    return s


@st.cache_data(show_spinner=False)
def prepare_model_data(
    monthly_df: pd.DataFrame,
    selected_labels: List[str],
    transform_map: Dict[str, str],
    start_date: str,
    end_date: str,
    min_non_null_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if monthly_df.empty or not selected_labels:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = monthly_df.loc[:, selected_labels].copy()
    df = df.loc[start_date:end_date]

    coverage = pd.DataFrame({
        "Series": df.columns,
        "Non-null count": [int(df[c].notna().sum()) for c in df.columns],
        "Coverage %": [round(100 * df[c].notna().mean(), 1) if len(df) else 0 for c in df.columns],
        "Transform": [transform_map.get(c, "level") for c in df.columns],
    })

    keep_cols = coverage.loc[coverage["Coverage %"] >= min_non_null_ratio * 100, "Series"].tolist()
    df = df[keep_cols]
    if df.empty:
        return pd.DataFrame(), coverage, pd.DataFrame()

    transformed = pd.DataFrame(index=df.index)
    for col in df.columns:
        transformed[col] = apply_transform(df[col], transform_map.get(col, "level"))

    transformed = transformed.replace([np.inf, -np.inf], np.nan)
    transformed = transformed.dropna(how="any")

    stationarity_rows = []
    for col in transformed.columns:
        pval = np.nan
        try:
            if transformed[col].dropna().shape[0] >= 12:
                pval = adfuller(transformed[col].dropna(), autolag="AIC")[1]
        except Exception:
            pass
        stationarity_rows.append({
            "Series": col,
            "ADF p-value": None if pd.isna(pval) else round(float(pval), 4),
            "Likely stationary": None if pd.isna(pval) else bool(pval < 0.05),
        })

    stationarity = pd.DataFrame(stationarity_rows)
    return transformed, coverage, stationarity


@st.cache_data(show_spinner=False)
def run_granger(df: pd.DataFrame, max_lag: int, alpha: float) -> pd.DataFrame:
    rows = []
    if df.shape[0] < max_lag + 8 or df.shape[1] < 2:
        return pd.DataFrame(rows)

    for src in df.columns:
        for tgt in df.columns:
            if src == tgt:
                continue
            pair = df[[tgt, src]].dropna()
            if pair.shape[0] < max_lag + 8:
                continue
            try:
                result = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
                for lag in range(1, max_lag + 1):
                    pval = result[lag][0]["ssr_ftest"][1]
                    stat = result[lag][0]["ssr_ftest"][0]
                    rows.append({
                        "model": "Granger",
                        "source": src,
                        "target": tgt,
                        "lag": lag,
                        "p_value": float(pval),
                        "score": float(-np.log10(max(pval, 1e-12))),
                        "stat": float(stat),
                        "significant": bool(pval < alpha),
                    })
            except Exception:
                continue

    out = pd.DataFrame(rows)
    return out.sort_values(["significant", "p_value", "score"], ascending=[False, True, False]) if not out.empty else out


@st.cache_data(show_spinner=False)
def fit_var_summary(df: pd.DataFrame, maxlags: int) -> Tuple[Optional[object], pd.DataFrame]:
    if df.shape[0] < maxlags + 10 or df.shape[1] < 2:
        return None, pd.DataFrame()
    try:
        model = VAR(df)
        fitted = model.fit(maxlags=maxlags, ic="aic")
        lag_order = fitted.k_ar
        rows = []
        for target in df.columns:
            for source in df.columns:
                if source == target:
                    continue
                for lag in range(1, lag_order + 1):
                    coef_name = f"L{lag}.{source}"
                    coef = fitted.params.loc[coef_name, target] if coef_name in fitted.params.index else np.nan
                    rows.append({
                        "target": target,
                        "source": source,
                        "lag": lag,
                        "coefficient": None if pd.isna(coef) else round(float(coef), 6),
                    })
        return fitted, pd.DataFrame(rows)
    except Exception:
        return None, pd.DataFrame()


@st.cache_data(show_spinner=False)
def run_pcmci(df: pd.DataFrame, tau_max: int, alpha: float, ci_test_name: str) -> pd.DataFrame:
    if not TIGRAMITE_AVAILABLE or df.shape[0] < tau_max + 8 or df.shape[1] < 2:
        return pd.DataFrame()

    z = (df - df.mean()) / df.std(ddof=0)
    z = z.dropna()
    if z.shape[0] < tau_max + 8:
        return pd.DataFrame()

    try:
        tg_df = TigramiteDataFrame(z.values, var_names=list(z.columns))
        ci_test = ParCorr(significance="analytic")
        if ci_test_name == "GPDC" and GPDC_AVAILABLE:
            ci_test = GPDC()

        pcmci = PCMCI(dataframe=tg_df, cond_ind_test=ci_test)
        res = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)
        pmat = res["p_matrix"]
        valmat = res.get("val_matrix")

        rows = []
        for tgt_idx, tgt in enumerate(z.columns):
            for src_idx, src in enumerate(z.columns):
                for lag in range(1, tau_max + 1):
                    pval = pmat[tgt_idx, src_idx, lag]
                    if np.isnan(pval):
                        continue
                    val = None if valmat is None else float(valmat[tgt_idx, src_idx, lag])
                    rows.append({
                        "model": "PCMCI",
                        "source": src,
                        "target": tgt,
                        "lag": lag,
                        "p_value": float(pval),
                        "score": float(-np.log10(max(pval, 1e-12))),
                        "effect": None if val is None else round(val, 6),
                        "significant": bool(pval < alpha),
                    })
        out = pd.DataFrame(rows)
        return out.sort_values(["significant", "p_value", "score"], ascending=[False, True, False]) if not out.empty else out
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def build_network_edges(results: pd.DataFrame, only_significant: bool) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "best_lag", "min_p_value"])
    df = results.copy()
    if only_significant:
        df = df[df["significant"]]
    if df.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "best_lag", "min_p_value"])

    agg = (
        df.sort_values("p_value")
        .groupby(["source", "target"], as_index=False)
        .first()[["source", "target", "score", "lag", "p_value"]]
        .rename(columns={"score": "weight", "lag": "best_lag", "p_value": "min_p_value"})
        .sort_values(["weight", "min_p_value"], ascending=[False, True])
    )
    return agg


st.title("Causal Macro Dashboard")
st.caption("Monthly-harmonized pipeline for Granger, VAR, and PCMCI analysis.")

with st.sidebar:
    st.header("Data")
    data_dir = st.text_input("Folder containing CSV files", value="/mnt/data")

    st.header("Model setup")
    default_series = [s.label for s in SERIES_SPECS if s.enabled_by_default]
    selected_series = st.multiselect(
        "Variables",
        options=[s.label for s in SERIES_SPECS],
        default=default_series,
    )

    transform_map = {}
    st.markdown("**Transform per variable**")
    for spec in SERIES_SPECS:
        if spec.label in selected_series:
            transform_map[spec.label] = st.selectbox(
                f"{spec.label}",
                TRANSFORM_OPTIONS,
                index=TRANSFORM_OPTIONS.index(spec.default_transform),
                key=f"transform_{spec.label}",
            )

    granger_alpha = st.slider("Granger significance level", 0.01, 0.20, 0.05, 0.01)
    granger_max_lag = st.slider("Granger max lag", 1, 12, 3)
    var_max_lag = st.slider("VAR max lag", 1, 12, 3)
    pcmci_enabled = st.checkbox("Run PCMCI", value=False)
    pcmci_tau = st.slider("PCMCI tau max", 1, 12, 3)
    pcmci_alpha = st.slider("PCMCI significance level", 0.01, 0.20, 0.05, 0.01)
    pcmci_test = st.selectbox(
        "PCMCI conditional independence test",
        ["ParCorr"] + (["GPDC"] if GPDC_AVAILABLE else []),
    )

    st.header("Sample window")
    min_coverage_ratio = st.slider("Min in-window coverage per series", 0.30, 1.00, 0.80, 0.05)
    only_significant = st.checkbox("Only show significant edges", value=True)

monthly_df, metadata_df = load_monthly_series(data_dir)

if monthly_df.empty:
    st.error("No valid series were loaded. Check the folder path and file names.")
    st.stop()

st.subheader("1) Data audit")
col_a, col_b = st.columns([1.2, 1.8])
with col_a:
    st.dataframe(metadata_df, use_container_width=True)
with col_b:
    st.write(f"Monthly panel before filtering: **{monthly_df.shape[0]} rows × {monthly_df.shape[1]} columns**")
    if not monthly_df.empty:
        overall_start = monthly_df.index.min().date()
        overall_end = monthly_df.index.max().date()
        start_date = st.date_input("Analysis start", value=max(pd.Timestamp("2021-01-31").date(), overall_start), min_value=overall_start, max_value=overall_end)
        end_date = st.date_input("Analysis end", value=min(pd.Timestamp("2024-05-31").date(), overall_end), min_value=overall_start, max_value=overall_end)
    else:
        st.stop()

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

model_df, coverage_df, stationarity_df = prepare_model_data(
    monthly_df,
    selected_series,
    transform_map,
    str(start_date),
    str(end_date),
    min_coverage_ratio,
)

st.subheader("2) Prepared modeling dataset")
left, right = st.columns([1.3, 1.7])
with left:
    st.dataframe(coverage_df, use_container_width=True)
    st.dataframe(stationarity_df, use_container_width=True)
with right:
    st.write(f"Model-ready sample: **{model_df.shape[0]} rows × {model_df.shape[1]} columns**")
    if not model_df.empty:
        st.line_chart(model_df, height=320)
    else:
        st.warning("No rows remain after filtering and transformations. Reduce the variable count, lower the coverage threshold, or adjust transforms.")
        st.stop()

st.subheader("3) Granger causality")
granger_results = run_granger(model_df, granger_max_lag, granger_alpha)
g_edges = build_network_edges(granger_results, only_significant)

g1, g2 = st.columns([1.1, 1.4])
with g1:
    st.write(f"Tests run: **{0 if granger_results.empty else len(granger_results)}**")
    st.write(f"Edges shown: **{len(g_edges)}**")
    st.dataframe(granger_results.head(25), use_container_width=True)
with g2:
    if ECHARTS_AVAILABLE and not g_edges.empty:
        option = {
            "tooltip": {},
            "series": [{
                "type": "graph",
                "layout": "force",
                "roam": True,
                "data": [{"name": c} for c in model_df.columns],
                "links": [
                    {
                        "source": row.source,
                        "target": row.target,
                        "value": round(float(row.weight), 2),
                        "label": {"show": True, "formatter": f"lag {int(row.best_lag)}"},
                    }
                    for row in g_edges.itertuples(index=False)
                ],
                "force": {"repulsion": 220, "edgeLength": 140},
                "emphasis": {"focus": "adjacency"},
                "lineStyle": {"curveness": 0.15},
            }],
        }
        st_echarts(options=option, height="420px")
    elif not g_edges.empty:
        st.info("Install streamlit-echarts to view the network graph.")
    else:
        st.info("No Granger edges found under the current settings.")

st.subheader("4) VAR summary")
var_model, var_coefs = fit_var_summary(model_df, var_max_lag)
if var_model is None:
    st.info("VAR could not be fit with the current sample size / settings.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected lag order", var_model.k_ar)
    c2.metric("AIC", round(var_model.aic, 4))
    c3.metric("BIC", round(var_model.bic, 4))
    st.dataframe(var_coefs.head(30), use_container_width=True)

if pcmci_enabled:
    st.subheader("5) PCMCI")
    if not TIGRAMITE_AVAILABLE:
        st.warning("Tigramite is not installed, so PCMCI cannot run in this environment.")
    else:
        pcmci_results = run_pcmci(model_df, pcmci_tau, pcmci_alpha, pcmci_test)
        p_edges = build_network_edges(pcmci_results, only_significant)
        p1, p2 = st.columns([1.1, 1.4])
        with p1:
            st.write(f"Tests run: **{0 if pcmci_results.empty else len(pcmci_results)}**")
            st.write(f"Edges shown: **{len(p_edges)}**")
            st.dataframe(pcmci_results.head(25), use_container_width=True)
        with p2:
            if ECHARTS_AVAILABLE and not p_edges.empty:
                option = {
                    "tooltip": {},
                    "series": [{
                        "type": "graph",
                        "layout": "force",
                        "roam": True,
                        "data": [{"name": c} for c in model_df.columns],
                        "links": [
                            {
                                "source": row.source,
                                "target": row.target,
                                "value": round(float(row.weight), 2),
                                "label": {"show": True, "formatter": f"lag {int(row.best_lag)}"},
                            }
                            for row in p_edges.itertuples(index=False)
                        ],
                        "force": {"repulsion": 220, "edgeLength": 140},
                        "emphasis": {"focus": "adjacency"},
                        "lineStyle": {"curveness": 0.15},
                    }],
                }
                st_echarts(options=option, height="420px")
            elif not p_edges.empty:
                st.info("Install streamlit-echarts to view the network graph.")
            else:
                st.info("No PCMCI edges found under the current settings.")

st.subheader("6) Download prepared data")
prepared_csv = model_df.reset_index().to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download model-ready dataset",
    data=prepared_csv,
    file_name="causal_model_ready_data.csv",
    mime="text/csv",
)
