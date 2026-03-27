import io
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


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
                "Start": None, "End": None, "Observations": 0,
            })
            continue
        if spec.column not in raw.columns:
            meta_rows.append({
                "Series": spec.label,
                "Status": f"Missing column: {spec.column}",
                "Start": None, "End": None, "Observations": 0,
            })
            continue

        df = raw[["date", spec.column]].copy()
        df[spec.column] = pd.to_numeric(df[spec.column], errors="coerce")
        df = df.dropna(subset=[spec.column]).set_index("date").sort_index()

        if spec.source_freq == "daily":
            df = df.resample("ME").mean() if spec.monthly_rule == "mean" else df.resample("ME").last()
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
    if transform == "diff1":
        return s.diff()
    if transform == "log":
        return np.log(s.where(s > 0))
    if transform == "log_diff1":
        return np.log(s.where(s > 0)).diff()
    if transform == "pct_change":
        return s.pct_change()
    return s  # level


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

    transformed = transformed.replace([np.inf, -np.inf], np.nan).dropna(how="any")

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

    return transformed, coverage, pd.DataFrame(stationarity_rows)


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
def fit_var_and_summarize(df: pd.DataFrame, maxlags: int) -> Tuple[Dict, pd.DataFrame]:
    """Fit VAR and return only serializable metrics + coefficient table (with p-values)."""
    if df.shape[0] < maxlags + 10 or df.shape[1] < 2:
        return {}, pd.DataFrame()
    try:
        fitted = VAR(df).fit(maxlags=maxlags, ic="aic")
        lag_order = fitted.k_ar
        metrics = {
            "k_ar": int(lag_order),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "hqic": float(fitted.hqic),
            "n_obs": int(fitted.nobs),
        }
        rows = []
        for target in df.columns:
            for source in df.columns:
                if source == target:
                    continue
                for lag in range(1, lag_order + 1):
                    key = f"L{lag}.{source}"
                    coef = fitted.params.loc[key, target] if key in fitted.params.index else np.nan
                    pval = fitted.pvalues.loc[key, target] if key in fitted.pvalues.index else np.nan
                    rows.append({
                        "target": target,
                        "source": source,
                        "lag": lag,
                        "coefficient": None if pd.isna(coef) else round(float(coef), 6),
                        "p_value": None if pd.isna(pval) else round(float(pval), 4),
                        "significant": None if pd.isna(pval) else bool(pval < 0.05),
                    })
        return metrics, pd.DataFrame(rows)
    except Exception:
        return {}, pd.DataFrame()


@st.cache_data(show_spinner=False)
def compute_irf(df: pd.DataFrame, maxlags: int, periods: int) -> Optional[bytes]:
    """Fit VAR and return orthogonalised IRF grid as PNG bytes."""
    if df.shape[0] < maxlags + 10 or df.shape[1] < 2:
        return None
    try:
        fitted = VAR(df).fit(maxlags=maxlags, ic="aic")
        irf_data = fitted.irf(periods=periods).orth_irfs  # (periods+1, n, n)
        n = df.shape[1]
        fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 2.2 * n), sharex=True)
        fig.suptitle("Orthogonalised IRF — rows: response variable, columns: shock variable", fontsize=10, y=1.01)
        for row in range(n):
            for col in range(n):
                ax = axes[row][col] if n > 1 else axes
                ax.plot(irf_data[:, row, col], color="steelblue", lw=1.5)
                ax.axhline(0, color="black", lw=0.7, linestyle="--")
                ax.tick_params(labelsize=6)
                if row == 0:
                    ax.set_title(df.columns[col], fontsize=8)
                if col == 0:
                    ax.set_ylabel(df.columns[row], fontsize=7, rotation=30, ha="right", labelpad=30)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def compute_fevd(df: pd.DataFrame, maxlags: int, periods: int) -> Optional[pd.DataFrame]:
    """Return FEVD as a long DataFrame: (target, source, horizon, fraction)."""
    if df.shape[0] < maxlags + 10 or df.shape[1] < 2:
        return None
    try:
        fitted = VAR(df).fit(maxlags=maxlags, ic="aic")
        decomp = fitted.fevd(periods=periods).decomp  # (n_vars, periods, n_vars)
        rows = []
        for tgt_idx, tgt in enumerate(df.columns):
            for h in range(1, periods + 1):
                for src_idx, src in enumerate(df.columns):
                    rows.append({
                        "target": tgt,
                        "source": src,
                        "horizon": h,
                        "fraction": round(float(decomp[tgt_idx, h - 1, src_idx]), 4),
                    })
        return pd.DataFrame(rows)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def run_pcmci(df: pd.DataFrame, tau_max: int, alpha: float, ci_test_name: str) -> pd.DataFrame:
    if not TIGRAMITE_AVAILABLE or df.shape[0] < tau_max + 8 or df.shape[1] < 2:
        return pd.DataFrame()

    z = ((df - df.mean()) / df.std(ddof=0)).dropna()
    if z.shape[0] < tau_max + 8:
        return pd.DataFrame()

    try:
        tg_df = TigramiteDataFrame(z.values, var_names=list(z.columns))
        ci_test = GPDC() if (ci_test_name == "GPDC" and GPDC_AVAILABLE) else ParCorr(significance="analytic")
        res = PCMCI(dataframe=tg_df, cond_ind_test=ci_test).run_pcmci(tau_max=tau_max, pc_alpha=alpha)
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
    empty = pd.DataFrame(columns=["source", "target", "weight", "best_lag", "min_p_value"])
    if results.empty:
        return empty
    df = results[results["significant"]].copy() if only_significant else results.copy()
    if df.empty:
        return empty
    return (
        df.sort_values("p_value")
        .groupby(["source", "target"], as_index=False)
        .first()[["source", "target", "score", "lag", "p_value"]]
        .rename(columns={"score": "weight", "lag": "best_lag", "p_value": "min_p_value"})
        .sort_values(["weight", "min_p_value"], ascending=[False, True])
    )


def render_causal_graph(edges: pd.DataFrame, nodes: List[str], height: str = "440px") -> None:
    """Directed force graph with arrows, node sizing by in-degree."""
    if not ECHARTS_AVAILABLE:
        st.info("Install streamlit-echarts to view the network graph.")
        return
    if edges.empty:
        st.info("No edges found under the current settings.")
        return

    in_deg = edges.groupby("target").size().to_dict()
    max_deg = max(in_deg.values()) if in_deg else 1

    node_data = [
        {
            "name": n,
            "symbolSize": 18 + 28 * in_deg.get(n, 0) / max_deg,
            "label": {"show": True, "fontSize": 10},
            "itemStyle": {"color": "#4e79a7" if n in in_deg else "#76b7b2"},
        }
        for n in nodes
    ]
    link_data = [
        {
            "source": r.source,
            "target": r.target,
            "value": round(float(r.weight), 2),
            "label": {"show": True, "formatter": f"lag {int(r.best_lag)}", "fontSize": 9},
        }
        for r in edges.itertuples(index=False)
    ]
    st_echarts(
        options={
            "tooltip": {"formatter": "{b}"},
            "series": [{
                "type": "graph",
                "layout": "force",
                "roam": True,
                "data": node_data,
                "links": link_data,
                "edgeSymbol": ["none", "arrow"],
                "edgeSymbolSize": [0, 10],
                "force": {"repulsion": 280, "edgeLength": 160, "gravity": 0.05},
                "emphasis": {"focus": "adjacency"},
                "lineStyle": {"curveness": 0.2, "color": "source", "width": 1.5},
            }],
        },
        height=height,
    )


# ── Layout ──────────────────────────────────────────────────────────────────

st.title("Causal Macro Dashboard")
st.caption("Vietnam macro data — monthly-harmonised pipeline for Granger, VAR (IRF / FEVD), and PCMCI causal analysis.")

with st.sidebar:
    st.header("Data")
    data_dir = st.text_input("Folder containing CSV files", value=_DEFAULT_DATA_DIR)

    st.header("Variables")
    selected_series = st.multiselect(
        "Select series",
        options=[s.label for s in SERIES_SPECS],
        default=[s.label for s in SERIES_SPECS if s.enabled_by_default],
    )

    st.markdown("**Transform per variable**")
    transform_map: Dict[str, str] = {}
    for spec in SERIES_SPECS:
        if spec.label in selected_series:
            transform_map[spec.label] = st.selectbox(
                spec.label,
                TRANSFORM_OPTIONS,
                index=TRANSFORM_OPTIONS.index(spec.default_transform),
                key=f"t_{spec.label}",
            )

    st.header("Model parameters")
    granger_alpha = st.slider("Granger α", 0.01, 0.20, 0.05, 0.01)
    granger_max_lag = st.slider("Granger max lag", 1, 12, 3)
    var_max_lag = st.slider("VAR max lag", 1, 12, 3)
    irf_periods = st.slider("IRF / FEVD horizon (months)", 6, 24, 12)

    pcmci_enabled = st.checkbox("Run PCMCI", value=False)
    if pcmci_enabled:
        pcmci_tau = st.slider("PCMCI tau max", 1, 12, 3)
        pcmci_alpha = st.slider("PCMCI α", 0.01, 0.20, 0.05, 0.01)
        pcmci_test = st.selectbox(
            "Conditional independence test",
            ["ParCorr"] + (["GPDC"] if GPDC_AVAILABLE else []),
        )
    else:
        pcmci_tau, pcmci_alpha, pcmci_test = 3, 0.05, "ParCorr"

    st.header("Filters")
    min_coverage_ratio = st.slider("Min coverage per series", 0.30, 1.00, 0.80, 0.05)
    only_significant = st.checkbox("Only show significant edges", value=True)

# ── 1) Data audit ────────────────────────────────────────────────────────────

monthly_df, metadata_df = load_monthly_series(data_dir)

if monthly_df.empty:
    st.error("No valid series loaded. Check the folder path and file names.")
    st.stop()

st.subheader("1) Data audit")
col_a, col_b = st.columns([1.2, 1.8])
with col_a:
    st.dataframe(metadata_df, use_container_width=True)
with col_b:
    st.write(f"Monthly panel: **{monthly_df.shape[0]} rows × {monthly_df.shape[1]} columns**")
    overall_start = monthly_df.index.min().date()
    overall_end = monthly_df.index.max().date()
    start_date = st.date_input(
        "Analysis start",
        value=max(pd.Timestamp("2021-01-31").date(), overall_start),
        min_value=overall_start, max_value=overall_end,
    )
    end_date = st.date_input(
        "Analysis end",
        value=min(pd.Timestamp("2024-05-31").date(), overall_end),
        min_value=overall_start, max_value=overall_end,
    )

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

model_df, coverage_df, stationarity_df = prepare_model_data(
    monthly_df, selected_series, transform_map,
    str(start_date), str(end_date), min_coverage_ratio,
)

# ── 2) Prepared dataset ──────────────────────────────────────────────────────

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
        st.warning("No rows remain after filtering. Reduce variables, lower the coverage threshold, or adjust transforms.")
        st.stop()

# ── 3) Correlation matrix ────────────────────────────────────────────────────

st.subheader("3) Correlation matrix")
corr = model_df.corr().round(3)
st.dataframe(
    corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.3f}"),
    use_container_width=True,
)
st.caption("Pearson correlation on the transformed series. Strong off-diagonal values indicate potential multicollinearity.")

# ── 4) Granger causality ─────────────────────────────────────────────────────

st.subheader("4) Granger causality")
granger_results = run_granger(model_df, granger_max_lag, granger_alpha)
g_edges = build_network_edges(granger_results, only_significant)

g1, g2 = st.columns([1.1, 1.4])
with g1:
    st.write(f"Tests run: **{len(granger_results)}**  |  Edges shown: **{len(g_edges)}**")
    st.dataframe(granger_results.head(30), use_container_width=True)
with g2:
    render_causal_graph(g_edges, list(model_df.columns))

# ── 5) VAR model ─────────────────────────────────────────────────────────────

st.subheader("5) VAR model")
var_metrics, var_coefs = fit_var_and_summarize(model_df, var_max_lag)

if not var_metrics:
    st.info("VAR could not be fit — check sample size or variable selection.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lag order (AIC)", var_metrics["k_ar"])
    c2.metric("AIC", round(var_metrics["aic"], 3))
    c3.metric("BIC", round(var_metrics["bic"], 3))
    c4.metric("Observations", var_metrics["n_obs"])
    with st.expander("Coefficient table (cross-variable lags only)", expanded=False):
        st.dataframe(var_coefs, use_container_width=True)

# ── 6) Impulse Response Functions ────────────────────────────────────────────

st.subheader("6) Impulse Response Functions (VAR)")
if not var_metrics:
    st.info("VAR model not available.")
else:
    with st.spinner("Computing IRFs…"):
        irf_png = compute_irf(model_df, var_max_lag, irf_periods)
    if irf_png:
        st.image(
            irf_png,
            caption=f"Orthogonalised IRF over {irf_periods} months. Each cell shows how a one-std shock to the column variable affects the row variable.",
            use_container_width=True,
        )
    else:
        st.warning("IRF computation failed.")

# ── 7) Forecast Error Variance Decomposition ─────────────────────────────────

st.subheader("7) Forecast Error Variance Decomposition (FEVD)")
if not var_metrics:
    st.info("VAR model not available.")
else:
    with st.spinner("Computing FEVD…"):
        fevd_df = compute_fevd(model_df, var_max_lag, irf_periods)
    if fevd_df is not None:
        selected_h = st.select_slider(
            "Horizon (months)",
            options=sorted(fevd_df["horizon"].unique()),
            value=min(12, int(fevd_df["horizon"].max())),
        )
        pivot = (
            fevd_df[fevd_df["horizon"] == selected_h]
            .pivot(index="target", columns="source", values="fraction")
            .round(3)
        )
        st.dataframe(
            pivot.style.background_gradient(cmap="Blues", vmin=0, vmax=1).format("{:.1%}"),
            use_container_width=True,
        )
        st.caption("Rows: variable being decomposed. Columns: shock source. Each row sums to 100%.")
    else:
        st.warning("FEVD computation failed.")

# ── 8) PCMCI ─────────────────────────────────────────────────────────────────

if pcmci_enabled:
    st.subheader("8) PCMCI")
    if not TIGRAMITE_AVAILABLE:
        st.warning("Tigramite is not installed — PCMCI cannot run.")
    else:
        with st.spinner("Running PCMCI…"):
            pcmci_results = run_pcmci(model_df, pcmci_tau, pcmci_alpha, pcmci_test)
        p_edges = build_network_edges(pcmci_results, only_significant)
        p1, p2 = st.columns([1.1, 1.4])
        with p1:
            st.write(f"Tests run: **{len(pcmci_results)}**  |  Edges shown: **{len(p_edges)}**")
            st.dataframe(pcmci_results.head(25), use_container_width=True)
        with p2:
            render_causal_graph(p_edges, list(model_df.columns))

# ── 9) Export ─────────────────────────────────────────────────────────────────

st.subheader("9) Export")
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "Download model-ready dataset (CSV)",
        data=model_df.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="causal_model_ready_data.csv",
        mime="text/csv",
    )
with dl2:
    if not granger_results.empty:
        st.download_button(
            "Download Granger results (CSV)",
            data=granger_results.to_csv(index=False).encode("utf-8"),
            file_name="granger_results.csv",
            mime="text/csv",
        )
