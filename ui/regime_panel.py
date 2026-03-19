"""
Regime probability panel UI components.
"""

import streamlit as st
import numpy as np
import pandas as pd
from config.settings import REGIME_NAMES, REGIME_COLORS, REGIME_ICONS
from core.directional_engine import DirectionalForecast
from ui.charts import plot_regime_probability_bars, plot_directional_gauge, plot_regime_timeline


def render_regime_header(forecast: DirectionalForecast, ticker: str, timeframe: str):
    """Top strip: current regime badge + key stats."""
    r = forecast.current_regime
    color = REGIME_COLORS[r]
    icon = REGIME_ICONS[r]
    name = REGIME_NAMES[r]

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(10,22,40,0.9), rgba(5,10,18,0.9));
        border: 1px solid {color}44;
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 16px;
        box-shadow: 0 0 20px {color}22;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-family:'Roboto Mono',monospace; font-size:0.7rem; color:#4a6b8a; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:4px;">
                    ◈ ACTIVE REGIME · {ticker} · {timeframe}
                </div>
                <div style="font-family:'Roboto Mono',monospace; font-size:1.4rem; color:{color}; font-weight:500; letter-spacing:0.05em;">
                    {icon} &nbsp; {name}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem; color:#4a6b8a; text-transform:uppercase; letter-spacing:0.12em;">Confidence</div>
                <div style="font-family:'Roboto Mono',monospace; font-size:1.8rem; color:#ffaa00; font-weight:500;">
                    {forecast.confidence:.1f}<span style="font-size:0.9rem;">%</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_regime_probability_panel(proba: np.ndarray):
    """7-regime probability bars."""
    st.markdown(
        '<div class="terminal-header">▸ REGIME POSTERIORS</div>',
        unsafe_allow_html=True,
    )
    fig = plot_regime_probability_bars(proba, height=300)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_directional_gauge(forecast: DirectionalForecast):
    """Directional probability gauge."""
    st.markdown(
        '<div class="terminal-header">▸ DIRECTIONAL BIAS</div>',
        unsafe_allow_html=True,
    )
    fig = plot_directional_gauge(
        forecast.p_up, forecast.p_down, forecast.confidence, height=260
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    # P(flat) and P(high vol)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("P(Flat)", f"{forecast.p_flat*100:.1f}%")
    with c2:
        st.metric("P(High Vol)", f"{forecast.p_high_vol*100:.1f}%")


def render_regime_stats_table(regime_stats: pd.DataFrame):
    """Regime statistics table."""
    st.markdown(
        '<div class="terminal-header">▸ REGIME STATISTICS</div>',
        unsafe_allow_html=True,
    )
    display = regime_stats[["name", "frequency", "mean_return", "volatility", "avg_duration_bars"]].copy()
    display.columns = ["Regime", "Frequency", "Mean Ret", "Ann. Vol", "Avg Bars"]
    display["Frequency"] = display["Frequency"].apply(lambda x: f"{x*100:.1f}%")
    display["Mean Ret"] = display["Mean Ret"].apply(lambda x: f"{x*100:.3f}%")
    display["Ann. Vol"] = display["Ann. Vol"].apply(lambda x: f"{x*100:.1f}%")
    display["Avg Bars"] = display["Avg Bars"].apply(lambda x: f"{x:.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_regime_timeline(tagged_df: pd.DataFrame):
    """Colored regime timeline."""
    st.markdown(
        '<div class="terminal-header">▸ REGIME HISTORY</div>',
        unsafe_allow_html=True,
    )
    fig = plot_regime_timeline(tagged_df, height=90)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_transition_stats(forecast: DirectionalForecast, persistence_bars: int):
    """Small transition metadata display."""
    r = forecast.current_regime
    color = REGIME_COLORS[r]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "P(Regime Shift)",
            f"{forecast.p_transition*100:.1f}%",
            help="Probability of transitioning to a different regime next bar",
        )
    with c2:
        st.metric(
            "Bars in Regime",
            f"{persistence_bars}",
            help="How many consecutive bars in current regime",
        )
    with c3:
        next_regime = "-"
        n_states = len(forecast.regime_proba)
        st.metric("Current Regime #", f"{r + 1} / {n_states}")
