"""
Scenario simulation panel UI components.
"""

import streamlit as st
import numpy as np
from config.settings import SCENARIOS, REGIME_NAMES, REGIME_COLORS, REGIME_ICONS
from core.scenario_simulator import ScenarioSimulator, ScenarioResult
from ui.charts import plot_scenario_comparison


def render_scenario_panel(
    simulator: ScenarioSimulator,
    X_full: np.ndarray,
    feature_names: list,
    directional_engine,
    transition_matrix: np.ndarray,
):
    """Full scenario simulation panel."""

    st.markdown(
        '<div class="terminal-header">▸ MARKET SCENARIO SIMULATOR</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-family:\'Roboto Mono\',monospace; font-size:0.72rem; color:#8baac9; margin-bottom:16px;">'
        '› Shock the current feature vector and observe how HMM regime probabilities shift under hypothetical market events.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Preset scenario buttons ───────────────────────────────────────────
    st.markdown('<div class="terminal-header">PRESET SCENARIOS</div>', unsafe_allow_html=True)

    scenario_names = list(SCENARIOS.keys())
    intensity = st.slider(
        "Shock Intensity",
        min_value=0.5, max_value=3.0, value=1.0, step=0.25,
        help="Multiplier on scenario shock amplitudes",
        key="scenario_intensity",
    )

    # 7 scenario buttons in a grid
    cols = st.columns(4)
    selected_scenario = st.session_state.get("selected_scenario", None)

    for idx, name in enumerate(scenario_names):
        cfg = SCENARIOS[name]
        col = cols[idx % 4]
        with col:
            if st.button(f"{cfg['icon']} {name}", key=f"scenario_{name}", use_container_width=True):
                selected_scenario = name
                st.session_state["selected_scenario"] = name

    # ── Custom scenario sliders ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="terminal-header">CUSTOM SCENARIO</div>', unsafe_allow_html=True)
    
    with st.expander("⚙️ Custom Feature Shocks", expanded=False):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            custom_return = st.slider("Log Return Shock (σ)", -5.0, 5.0, 0.0, 0.5, key="c_ret")
            custom_vol = st.slider("Realized Vol Shock (σ)", -3.0, 5.0, 0.0, 0.5, key="c_vol")
        with cc2:
            custom_rsi = st.slider("RSI Shock (σ)", -3.0, 3.0, 0.0, 0.5, key="c_rsi")
            custom_macd = st.slider("MACD Hist Shock (σ)", -3.0, 3.0, 0.0, 0.5, key="c_macd")
        with cc3:
            custom_vol_z = st.slider("Volume Z-Score Shock (σ)", -3.0, 5.0, 0.0, 0.5, key="c_volz")
            custom_momentum = st.slider("Momentum Shock (σ)", -3.0, 3.0, 0.0, 0.5, key="c_mom")
        
        if st.button("▶ RUN CUSTOM SCENARIO", key="run_custom_scenario"):
            custom_shocks = {}
            if custom_return != 0.0: custom_shocks["log_return"] = custom_return
            if custom_vol != 0.0: custom_shocks["realized_vol"] = custom_vol
            if custom_rsi != 0.0: custom_shocks["rsi"] = custom_rsi
            if custom_macd != 0.0: custom_shocks["macd_hist"] = custom_macd
            if custom_vol_z != 0.0: custom_shocks["volume_zscore"] = custom_vol_z
            if custom_momentum != 0.0: custom_shocks["momentum"] = custom_momentum
            
            if custom_shocks:
                selected_scenario = "__custom__"
                st.session_state["selected_scenario"] = "__custom__"
                st.session_state["custom_shocks"] = custom_shocks

    # ── Run and display selected scenario results ─────────────────────────
    if selected_scenario:
        try:
            if selected_scenario == "__custom__":
                custom_shocks = st.session_state.get("custom_shocks", {})
                if custom_shocks:
                    result = simulator.run_custom_scenario(
                        X_full, feature_names, custom_shocks, intensity,
                        directional_engine, transition_matrix,
                    )
                    result.scenario_name = "Custom"
                else:
                    st.warning("Define at least one non-zero shock parameter.")
                    return
            else:
                result = simulator.run_scenario(
                    X_full, feature_names, selected_scenario, intensity,
                    directional_engine, transition_matrix,
                )
            _render_scenario_result(result)
        except Exception as e:
            st.error(f"Scenario simulation failed: {e}")


def _render_scenario_result(result: ScenarioResult):
    """Display scenario result: before/after comparison."""
    st.markdown("---")

    # Header
    regime_changed_str = "⚠️ REGIME CHANGE DETECTED" if result.regime_changed else "→ Regime Unchanged"
    change_color = "#ffaa00" if result.regime_changed else "#8baac9"

    st.markdown(f"""
    <div style="
        background: rgba(10,22,40,0.85);
        border: 1px solid rgba(0,170,255,0.2);
        border-left: 4px solid rgba(0,170,255,0.6);
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 16px;
    ">
        <div style="font-family:'Roboto Mono',monospace; font-size:0.7rem; color:#4a6b8a; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">
            SCENARIO RESULT · {result.icon} {result.scenario_name}
        </div>
        <div style="font-family:'Roboto Mono',monospace; font-size:0.85rem; color:#e8f4fd;">
            {result.description}
        </div>
        <div style="font-family:'Roboto Mono',monospace; font-size:0.8rem; color:{change_color}; margin-top:8px; font-weight:500;">
            {regime_changed_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Regime shift summary
    col1, col2, col3, col4 = st.columns(4)
    base_r = int(np.argmax(result.baseline_proba))
    shock_r = int(np.argmax(result.shocked_proba))
    with col1:
        st.metric("Baseline Regime", f"{REGIME_ICONS[base_r]} {REGIME_NAMES[base_r].split(' ')[0]}")
    with col2:
        st.metric("Shocked Regime", f"{REGIME_ICONS[shock_r]} {REGIME_NAMES[shock_r].split(' ')[0]}")
    with col3:
        delta_up = (result.p_up_shocked - result.p_up_baseline) * 100
        st.metric("P(↑) Change", f"{result.p_up_shocked*100:.1f}%", delta=f"{delta_up:+.1f}%")
    with col4:
        delta_down = (result.p_down_shocked - result.p_down_baseline) * 100
        st.metric("P(↓) Change", f"{result.p_down_shocked*100:.1f}%", delta=f"{delta_down:+.1f}%")

    # Comparison chart
    fig = plot_scenario_comparison(result.baseline_proba, result.shocked_proba, height=300)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Delta table
    st.markdown('<div class="terminal-header">▸ PROBABILITY SHIFTS</div>', unsafe_allow_html=True)
    delta_rows = []
    n_states = len(result.baseline_proba)
    for i, name in enumerate(REGIME_NAMES[:n_states]):
        delta = result.proba_delta[i] * 100
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        color = "#00ff88" if delta > 0 else ("#ff3366" if delta < 0 else "#8baac9")
        delta_rows.append({
            "Regime": f"{REGIME_ICONS[i]} {name}",
            "Baseline": f"{result.baseline_proba[i]*100:.2f}%",
            "Shocked": f"{result.shocked_proba[i]*100:.2f}%",
            "Δ": f"{arrow} {abs(delta):.2f}%",
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)
