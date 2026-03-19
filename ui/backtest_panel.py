"""
Backtesting panel UI components.
"""

import streamlit as st
import pandas as pd
from core.backtester import STRATEGY_DESCRIPTIONS
from ui.charts import plot_equity_curve


def render_backtest_panel(tagged_df: pd.DataFrame, backtester):
    """Full backtesting UI panel."""

    st.markdown(
        '<div class="terminal-header">▸ STRATEGY BACKTESTER</div>',
        unsafe_allow_html=True,
    )

    # Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        strategy = st.selectbox(
            "Strategy",
            options=list(STRATEGY_DESCRIPTIONS.keys()),
            format_func=lambda k: f"{k.replace('_', ' ').title()} — {STRATEGY_DESCRIPTIONS[k]}",
            key="backtest_strategy",
        )
    with c2:
        prob_threshold = st.slider(
            "Prob. Threshold",
            min_value=0.2, max_value=0.9, value=0.5, step=0.05,
            help="Minimum regime probability required to enter a trade",
            key="backtest_prob",
        )
    with c3:
        run_btn = st.button("▶ RUN BACKTEST", key="run_backtest")

    # Strategy description
    st.markdown(
        f'<div style="font-family:\'Roboto Mono\',monospace; font-size:0.72rem; '
        f'color:#8baac9; margin-bottom:12px;">› {STRATEGY_DESCRIPTIONS[strategy]}</div>',
        unsafe_allow_html=True,
    )

    if run_btn or st.session_state.get("backtest_result"):
        if run_btn or not st.session_state.get("backtest_result"):
            with st.spinner("Running backtest..."):
                result = backtester.run(tagged_df, strategy, prob_threshold)
                st.session_state["backtest_result"] = result
                st.session_state["backtest_strategy_name"] = strategy
        else:
            result = st.session_state["backtest_result"]

        _render_metrics(result)
        _render_equity_chart(result)
        _render_regime_pnl(result)


def _render_metrics(result: dict):
    """KPI metric cards."""
    m = result["metrics"]
    st.markdown('<div class="terminal-header" style="margin-top:12px;">▸ PERFORMANCE METRICS</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics_to_show = [
        (c1, "Sharpe Ratio", f"{m['sharpe']:.2f}", ""),
        (c2, "Max Drawdown", f"{m['max_drawdown']*100:.1f}%", "delta_color:inverse"),
        (c3, "Win Rate", f"{m['win_rate']*100:.1f}%", ""),
        (c4, "Profit Factor", f"{m['profit_factor']:.2f}", ""),
        (c5, "Net Return", f"{m['net_return']*100:.1f}%", ""),
        (c6, "Total Trades", f"{m['total_trades']}", ""),
    ]
    for col, label, val, _ in metrics_to_show:
        with col:
            st.metric(label, val)


def _render_equity_chart(result: dict):
    """Equity curve + drawdown chart."""
    st.markdown('<div class="terminal-header" style="margin-top:12px;">▸ EQUITY CURVE</div>', unsafe_allow_html=True)
    fig = plot_equity_curve(result, height=320)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


def _render_regime_pnl(result: dict):
    """Per-regime P&L breakdown."""
    regime_pnl = result.get("regime_pnl", {})
    if not regime_pnl:
        return

    st.markdown('<div class="terminal-header" style="margin-top:12px;">▸ PER-REGIME P&L BREAKDOWN</div>', unsafe_allow_html=True)
    rows = []
    for regime_name, stats in regime_pnl.items():
        rows.append({
            "Regime": regime_name,
            "Total Return": f"{stats['total_return']*100:.3f}%",
            "# Bars Active": stats['n_bars'],
            "Win Rate": f"{stats['win_rate']*100:.1f}%",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
