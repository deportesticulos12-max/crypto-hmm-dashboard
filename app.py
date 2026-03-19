"""
CryptoHMM — Regime Intelligence Terminal
Main Streamlit entry point.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import logging
import traceback

# ── Page Config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="CryptoHMM — Regime Intelligence Terminal",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────
from ui.styles import inject_css
inject_css()

from config.settings import (
    ASSET_LIST, ASSET_DISPLAY, TIMEFRAME_LIST, TIMEFRAME_BARS,
    HMM_N_STATES, REGIME_NAMES, REGIME_COLORS, REGIME_ICONS,
    APP_TITLE, REFRESH_INTERVAL_OPTIONS, DEFAULT_REFRESH,
    DEFAULT_INITIAL_CAPITAL, SCENARIOS,
)
from core.data_fetcher import DataManager
from core.feature_engineering import FeatureEngineer
from core.hmm_model import CryptoHMM
from core.regime_analyzer import RegimeAnalyzer
from core.directional_engine import DirectionalEngine
from core.backtester import RegimeBacktester, STRATEGY_DESCRIPTIONS
from core.scenario_simulator import ScenarioSimulator
from core.ensemble import EnsembleHMM
from streamlit_lightweight_charts import renderLightweightCharts
from ui.charts import (
    get_lightweight_chart_dict, plot_transition_matrix,
    plot_volatility_chart,
)
from ui.regime_panel import (
    render_regime_header, render_regime_probability_panel,
    render_directional_gauge, render_regime_stats_table,
    render_regime_timeline, render_transition_stats,
)
from ui.backtest_panel import render_backtest_panel
from ui.scenario_panel import render_scenario_panel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Session-State Init ─────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "fitted": False,
        "tagged_df": None,
        "X_full": None,
        "feature_names": [],
        "hmm": None,
        "fe": None,
        "directional_engine": None,
        "regime_analyzer": None,
        "backtester": None,
        "forecast": None,
        "regime_stats": None,
        "transition_matrix": None,
        "last_refresh": 0.0,
        "error_msg": None,
        "selected_scenario": None,
        "backtest_result": None,
        "use_ensemble": False,
        "ensemble_model": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Core Pipeline ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_data(symbol: str, interval: str, limit: int, use_cache: bool = True, source: str = "binance"):
    dm = DataManager()
    return dm.get_data(symbol, interval, limit=limit, use_cache=use_cache, source=source)


def run_pipeline(symbol: str, interval: str, n_states: int, limit: int, use_ensemble: bool = False, force: bool = False, source: str = "binance"):
    """Full pipeline: fetch → features → HMM → analyzer → directional."""
    try:
        with st.spinner(f"📡 Fetching market data ({limit} bars from {source})..."):
            df = _fetch_data(symbol, interval, limit, use_cache=not force, source=source)

        with st.spinner("⚙️ Engineering features..."):
            fe = FeatureEngineer()
            X, feature_names = fe.transform(df)
            
            # Ensure we take exactly the requested limit from the end (after warm-up loss)
            if X.shape[0] > limit:
                X = X[-limit:]
                
            if X.shape[0] < 50:
                st.session_state["error_msg"] = "Not enough data bars for training. Try a longer timeframe."
                return False

        with st.spinner(f"🧬 Training {n_states}-state HMM..."):
            if use_ensemble:
                hmm = EnsembleHMM(n_models=5, n_states=n_states)
                hmm.train(X)
                proba_all, _ = hmm.predict_proba(X)
                regimes = hmm.decode(X)
                st.session_state["ensemble_model"] = hmm
            else:
                hmm = CryptoHMM(n_states)
                hmm.train(X)
                proba_all = hmm.predict_proba(X)
                regimes = hmm.decode(X)

        with st.spinner("📊 Analyzing regimes..."):
            analyzer = RegimeAnalyzer(n_states)
            tagged_df = analyzer.tag_dataframe(df, regimes, proba_all)
            regime_stats = analyzer.compute_regime_stats(tagged_df)
            transition_matrix = hmm.get_transition_matrix()

        with st.spinner("🎯 Computing directional probabilities..."):
            de = DirectionalEngine(n_states)
            de.fit(tagged_df)
            latest_proba = proba_all[-1]
            recent_returns = np.log(
                tagged_df["close"] / tagged_df["close"].shift(1)
            ).fillna(0).values[-5:]
            forecast = de.forecast(latest_proba, transition_matrix, recent_returns)

        # Store in session state
        st.session_state.update({
            "fitted": True,
            "tagged_df": tagged_df,
            "X_full": X,
            "feature_names": feature_names,
            "hmm": hmm,
            "fe": fe,
            "directional_engine": de,
            "regime_analyzer": analyzer,
            "backtester": RegimeBacktester(DEFAULT_INITIAL_CAPITAL),
            "forecast": forecast,
            "regime_stats": regime_stats,
            "transition_matrix": transition_matrix,
            "last_refresh": time.time(),
            "error_msg": None,
            "use_ensemble": use_ensemble,
        })
        return True

    except Exception as e:
        st.session_state["error_msg"] = f"Pipeline error: {str(e)}\n{traceback.format_exc()}"
        st.session_state["fitted"] = False
        return False


# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="
            font-family: 'Roboto Mono', monospace;
            font-size: 0.65rem;
            color: #4a6b8a;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            margin-bottom: 4px;
        ">◈ CryptoHMM Terminal v1.0</div>
        <div style="
            font-family: 'Roboto Mono', monospace;
            font-size: 1.0rem;
            color: #00ff88;
            font-weight: 500;
            margin-bottom: 16px;
            letter-spacing: 0.05em;
        ">🧬 Regime Intelligence</div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Asset & Timeframe")
        symbol = st.selectbox(
            "Asset",
            options=ASSET_LIST,
            format_func=lambda s: ASSET_DISPLAY[s],
            key="sidebar_symbol",
            index=0,
        )
        interval = st.selectbox(
            "Timeframe",
            options=TIMEFRAME_LIST,
            index=TIMEFRAME_LIST.index("4h"),
            key="sidebar_interval",
        )

        st.markdown("### 📡 Data Source")
        source = st.radio(
            "Data Source",
            options=["binance", "yahoo"],
            format_func=lambda x: "Binance (Futures/Spot)" if x == "binance" else "Yahoo Finance (Long Hist)",
            index=0,
            key="sidebar_source",
            help="Yahoo is better for >1000 bars in daily/hourly timeframes."
        )

        st.markdown("### 🧬 Model Settings")
        n_states = st.selectbox(
            "HMM States",
            options=[4, 5, 7],
            index=2,
            key="sidebar_states",
            help="Number of hidden market regimes to detect."
        )
        
        limit_bars = st.selectbox(
            "Historical Data Size",
            options=[500, 1000, 1500, 2000, 3000],
            index=1,
            key="sidebar_limit",
            help="Number of historical candles to fetch for training."
        )

        use_ensemble = st.checkbox(
            "Use Ensemble HMM (5 models)",
            value=False,
            key="sidebar_ensemble",
            help="Train 5 HMMs with different seeds and average posteriors for higher stability",
        )

        st.markdown("### 🔄 Data Controls")
        use_live_cache = st.checkbox("Use cache", value=True, key="sidebar_cache")

        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button("▶ TRAIN MODEL", key="sidebar_run", use_container_width=True)
        with col2:
            refresh_btn = st.button("↻ REFRESH DATA", key="sidebar_refresh", use_container_width=True)

        if "last_refresh" in st.session_state and st.session_state["last_refresh"] > 0:
            elapsed = int(time.time() - st.session_state["last_refresh"])
            st.markdown(
                f'<div style="font-family:\'Roboto Mono\',monospace; font-size:0.65rem; '
                f'color:#4a6b8a;">Last refresh: {elapsed}s ago</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### ℹ️ Regime Legend")
        
        # Only show the relevant legend colors/names based on N states selected (we gracefully slice them)
        disp_colors = REGIME_COLORS[:n_states]
        disp_icons = REGIME_ICONS[:n_states]
        disp_names = REGIME_NAMES[:n_states]
        
        for i, (name, color, icon) in enumerate(zip(disp_names, disp_colors, disp_icons)):
            st.markdown(
                f'<div style="font-family:\'Roboto Mono\',monospace; font-size:0.72rem; '
                f'color:{color}; margin:2px 0;">{icon} {name}</div>',
                unsafe_allow_html=True,
            )

        return symbol, interval, n_states, limit_bars, use_ensemble, source, run_btn or refresh_btn


# ── Tabs ───────────────────────────────────────────────────────────────────

def render_dashboard_tab():
    """Main dashboard tab."""
    state = st.session_state

    if not state["fitted"]:
        _render_welcome()
        return

    forecast = state["forecast"]
    tagged_df = state["tagged_df"]
    regime_stats = state["regime_stats"]
    transition_matrix = state["transition_matrix"]
    analyzer = state["regime_analyzer"]

    # ── Regime header banner ──────────────────────────────────────────────
    render_regime_header(
        forecast,
        st.session_state.get("sidebar_symbol", "BTCUSDT"),
        st.session_state.get("sidebar_interval", "4h"),
    )

    # ── Top metrics row ───────────────────────────────────────────────────
    persistence_bars = analyzer.get_current_regime_bar_count(tagged_df)
    render_transition_stats(forecast, persistence_bars)

    st.markdown("---")

    # ── Row 1: Price chart (left wide) + regime probabilities (right) ─────
    col_chart, col_proba = st.columns([3, 1])

    with col_chart:
        chart_opts = get_lightweight_chart_dict(tagged_df)
        renderLightweightCharts(chart_opts, 'price_chart')

    with col_proba:
        render_regime_probability_panel(forecast.regime_proba)
        render_directional_gauge(forecast)

    # ── Regime timeline ───────────────────────────────────────────────────
    render_regime_timeline(tagged_df)

    st.markdown("---")

    # ── Row 2: Transition matrix + Volatility chart ───────────────────────
    col_trans, col_vol = st.columns(2)

    with col_trans:
        fig_trans = plot_transition_matrix(transition_matrix, height=380)
        st.plotly_chart(fig_trans, use_container_width=True, config={"displayModeBar": False})

    with col_vol:
        fig_vol = plot_volatility_chart(tagged_df, height=380)
        st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ── Row 3: Regime stats table ─────────────────────────────────────────
    render_regime_stats_table(regime_stats)


def render_model_info_tab():
    """Model information tab."""
    state = st.session_state

    if not state["fitted"]:
        st.info("Train the model first using the sidebar controls.")
        return

    hmm = state["hmm"]
    feature_names = state["feature_names"]
    tagged_df = state["tagged_df"]

    st.markdown('<div class="terminal-header">▸ HMM MODEL INFORMATION</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", "Gaussian HMM")
        st.metric("States", f"{HMM_N_STATES}")
        st.metric("Covariance", "Full")
    with col2:
        if state.get("use_ensemble"):
            st.metric("Log-Likelihood", f"{hmm.score(state['X_full']):.2f}")
        else:
            st.metric("Log-Likelihood", f"{hmm.log_likelihood:.2f}")
        st.metric("Features", f"{len(feature_names)}")
        st.metric("Training Bars", f"{len(tagged_df)}")
    with col3:
        st.metric("Mode", "Ensemble" if state.get("use_ensemble") else "Single HMM")
        st.metric("Viterbi States", "7")
        st.metric("Emissions", "Gaussian (full cov)")

    st.markdown("---")
    st.markdown('<div class="terminal-header">▸ FEATURE SET</div>', unsafe_allow_html=True)

    # Display feature names in a 3-column grid
    feat_cols = st.columns(3)
    for i, name in enumerate(feature_names):
        with feat_cols[i % 3]:
            st.markdown(
                f'<div style="font-family:\'Roboto Mono\',monospace; font-size:0.72rem; '
                f'color:#8baac9; padding:2px 0; border-bottom:1px solid rgba(0,170,255,0.05);">'
                f'• {name}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="terminal-header">▸ REGIME STATE MEANS (FIRST FEATURE = LOG RETURN)</div>', unsafe_allow_html=True)

    means = hmm.state_means
    means_data = []
    for i in range(hmm.n_states):
        means_data.append({
            "Regime": f"{REGIME_ICONS[i]} {REGIME_NAMES[i]}",
            "Mean Log Return": f"{means[i, 0]*100:.4f}%",
        })
    st.dataframe(pd.DataFrame(means_data), use_container_width=True, hide_index=True)


def _render_welcome():
    """Welcome screen when no model is trained."""
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <div style="font-size: 4rem; margin-bottom: 16px;">🧬</div>
        <div style="
            font-family: 'Roboto Mono', monospace;
            font-size: 1.8rem;
            color: #00ff88;
            font-weight: 500;
            margin-bottom: 12px;
        ">CryptoHMM — Regime Intelligence Terminal</div>
        <div style="
            font-family: 'Roboto Mono', monospace;
            font-size: 0.9rem;
            color: #8baac9;
            max-width: 600px;
            margin: 0 auto 32px auto;
            line-height: 1.7;
        ">
            Select an asset and timeframe in the sidebar,<br>
            then click <span style="color:#ffaa00;">▶ TRAIN MODEL</span> to begin regime detection.
        </div>
        <div style="
            display: flex;
            justify-content: center;
            gap: 24px;
            flex-wrap: wrap;
        ">
    """, unsafe_allow_html=True)

    for i, (name, color, icon) in enumerate(zip(REGIME_NAMES, REGIME_COLORS, REGIME_ICONS)):
        st.markdown(f"""
        <div style="
            background: rgba(10,22,40,0.7);
            border: 1px solid {color}33;
            border-left: 3px solid {color};
            border-radius: 6px;
            padding: 10px 16px;
            font-family: 'Roboto Mono', monospace;
            font-size: 0.78rem;
            color: {color};
            min-width: 200px;
        ">{icon} &nbsp; {name}</div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


# ── Main App ───────────────────────────────────────────────────────────────

def main():
    # Sidebar controls
    symbol, interval, n_states, limit_bars, use_ensemble, source, trigger = render_sidebar()

    # Trigger pipeline
    if trigger:
        # Clear backtest result on new data load
        st.session_state["backtest_result"] = None
        st.session_state["selected_scenario"] = None
        # Explicitly pass all variables to ensure the pipeline receives the UI selection
        success = run_pipeline(
            symbol=symbol, 
            interval=interval, 
            n_states=n_states, 
            limit=limit_bars, 
            use_ensemble=use_ensemble, 
            force=True,
            source=source
        )
        if not success and st.session_state.get("error_msg"):
            st.error(st.session_state["error_msg"])

    # Error banner
    if st.session_state.get("error_msg") and not st.session_state["fitted"]:
        st.error(st.session_state["error_msg"])

    # Main area — title bar
    tc, ts = st.columns([6, 1])
    with tc:
        st.markdown(
            f'<h1 style="font-family:\'Roboto Mono\',monospace; font-size:1.25rem; '
            f'color:#e8f4fd; margin:0; padding:0; font-weight:500; letter-spacing:0.05em;">'
            f'🧬 {APP_TITLE}</h1>',
            unsafe_allow_html=True,
        )
    with ts:
        if st.session_state["fitted"]:
            st.markdown(
                '<div style="font-family:\'Roboto Mono\',monospace; font-size:0.65rem; '
                'color:#00ff88; text-align:right; padding-top:8px;">● LIVE</div>',
                unsafe_allow_html=True,
            )

    # Tabs
    tab_dashboard, tab_backtest, tab_scenarios, tab_model = st.tabs([
        "📊 Dashboard",
        "📈 Backtest",
        "🧪 Scenarios",
        "⚙️ Model Info",
    ])

    with tab_dashboard:
        render_dashboard_tab()

    with tab_backtest:
        if st.session_state["fitted"]:
            render_backtest_panel(
                st.session_state["tagged_df"],
                st.session_state["backtester"],
            )
        else:
            st.info("Train the model first using the sidebar controls.")

    with tab_scenarios:
        if st.session_state["fitted"]:
            simulator = ScenarioSimulator(
                st.session_state["hmm"],
                st.session_state["fe"],
                n_states=n_states,
            )
            render_scenario_panel(
                simulator,
                st.session_state["X_full"],
                st.session_state["feature_names"],
                st.session_state["directional_engine"],
                st.session_state["transition_matrix"],
            )
        else:
            st.info("Train the model first using the sidebar controls.")

    with tab_model:
        render_model_info_tab()


if __name__ == "__main__":
    main()
