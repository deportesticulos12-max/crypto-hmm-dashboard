"""
All Plotly chart functions for the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config.settings import REGIME_NAMES, REGIME_COLORS, REGIME_ICONS

# ── Shared Layout Defaults ─────────────────────────────────────────────────

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(5, 10, 18, 0.0)",
    plot_bgcolor="rgba(5, 10, 18, 0.0)",
    font=dict(family="Roboto Mono, monospace", color="#8baac9", size=11),
    title_font=dict(family="Roboto Mono, monospace", color="#e8f4fd", size=13),
    legend=dict(
        bgcolor="rgba(10, 22, 40, 0.7)",
        bordercolor="rgba(0, 170, 255, 0.2)",
        borderwidth=1,
        font=dict(size=10),
    ),
    margin=dict(l=50, r=20, t=50, b=40),
    xaxis=dict(
        gridcolor="rgba(0, 170, 255, 0.08)",
        linecolor="rgba(0, 170, 255, 0.2)",
        tickfont=dict(size=9),
        showspikes=True,
        spikecolor="rgba(0, 255, 136, 0.4)",
        spikethickness=1,
    ),
    yaxis=dict(
        gridcolor="rgba(0, 170, 255, 0.08)",
        linecolor="rgba(0, 170, 255, 0.2)",
        tickfont=dict(size=9),
        showspikes=True,
        spikecolor="rgba(0, 255, 136, 0.4)",
        spikethickness=1,
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(10, 22, 40, 0.95)",
        bordercolor="rgba(0, 255, 136, 0.5)",
        font=dict(family="Roboto Mono", size=11, color="#e8f4fd"),
    ),
)


def _apply_dark_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(height=height, title=title, **_DARK_LAYOUT)
    return fig


# ── Price Chart with Regime Overlay ───────────────────────────────────────

def get_lightweight_chart_dict(tagged_df: pd.DataFrame) -> list:
    """Generates the dictionary payload for streamlit-lightweight-charts."""
    chart_options = {
        "layout": {
            "textColor": "#8baac9",
            "background": {"type": "solid", "color": "rgba(5, 10, 18, 0.0)"},
        },
        "grid": {
            "vertLines": {"color": "rgba(0, 170, 255, 0.08)"},
            "horzLines": {"color": "rgba(0, 170, 255, 0.08)"},
        },
        "crosshair": {
            "mode": 1,
            "vertLine": {"color": "rgba(0, 255, 136, 0.4)", "width": 1, "style": 1},
            "horzLine": {"color": "rgba(0, 255, 136, 0.4)", "width": 1, "style": 1},
        },
        "timeScale": {
            "timeVisible": True,
            "secondsVisible": False,
        },
        "rightPriceScale": {
            "borderColor": "rgba(0, 170, 255, 0.2)",
            "scaleMargins": {"top": 0.1, "bottom": 0.2},
        }
    }

    candle_data = []
    markers = []
    prev_regime = None

    for idx, (ts, row) in enumerate(tagged_df.iterrows()):
        t = int(ts.timestamp())
        candle_data.append({
            "time": t,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
        })
        
        if "regime" in tagged_df.columns:
            regime = int(row["regime"])
            if regime != prev_regime:
                color = REGIME_COLORS[regime]
                icon = REGIME_ICONS[regime]
                markers.append({
                    "time": t,
                    "position": "aboveBar",
                    "color": color,
                    "shape": "arrowDown",
                    "text": f"{icon} {REGIME_NAMES[regime].split('/')[0].strip()}",
                })
                prev_regime = regime

    candle_series = {
        "type": "Candlestick",
        "data": candle_data,
        "options": {
            "upColor": "#00ff88",
            "downColor": "#ff3366",
            "borderVisible": False,
            "wickUpColor": "#00ff88",
            "wickDownColor": "#ff3366",
        },
        "markers": markers,
    }

    volume_data = []
    for ts, row in tagged_df.iterrows():
        t = int(ts.timestamp())
        # Base color
        color = "#00ff88" if row["close"] >= row["open"] else "#ff3366"
        # If regime exists, mix opacity to show regime bands loosely in volume
        if "regime" in tagged_df.columns:
            rc = REGIME_COLORS[int(row["regime"])]
            color = _hex_to_rgba(rc, 0.7)
            
        volume_data.append({
            "time": t,
            "value": row["volume"],
            "color": color,
        })

    volume_series = {
        "type": "Histogram",
        "data": volume_data,
        "options": {
            "priceFormat": {"type": "volume"},
            "priceScaleId": "", # Set as overlay
            "scaleMargins": {"top": 0.8, "bottom": 0},
        }
    }

    return [
        {
            "chart": chart_options,
            "series": [volume_series, candle_series] 
        }
    ]


# ── Regime Probability Bar Chart ──────────────────────────────────────────

def plot_regime_probability_bars(proba: np.ndarray, height: int = 320) -> go.Figure:
    """Horizontal probability bars for N regimes."""
    n_states = len(proba)
    values = [float(p) * 100 for p in proba]
    labels = [f"{REGIME_ICONS[i]}  {REGIME_NAMES[i]}" for i in range(n_states)]
    colors = REGIME_COLORS[:n_states]

    fig = go.Figure()
    for i in range(n_states):
        fig.add_trace(go.Bar(
            y=[labels[i]],
            x=[values[i]],
            orientation="h",
            name=labels[i],
            marker_color=colors[i],
            marker_line_width=0,
            opacity=0.85 if values[i] == max(values) else 0.55,
            text=f"{values[i]:.1f}%",
            textposition="outside",
            textfont=dict(family="Roboto Mono", size=10, color="#e8f4fd"),
        ))

    fig.update_layout(
        height=height,
        title="Regime Probability Distribution",
        showlegend=False,
        xaxis=dict(
            range=[0, 105],
            ticksuffix="%",
            gridcolor="rgba(0, 170, 255, 0.08)",
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            tickfont=dict(family="Roboto Mono", size=10, color="#e8f4fd"),
            gridcolor="rgba(0,0,0,0)",
        ),
        bargap=0.3,
        **{k: v for k, v in _DARK_LAYOUT.items() if k not in ("hovermode", "xaxis", "yaxis")},
    )
    return fig


# ── Transition Matrix Heatmap ─────────────────────────────────────────────

def plot_transition_matrix(trans_matrix: np.ndarray, height: int = 420) -> go.Figure:
    """Interactive heatmap of the HMM transition matrix."""
    n_states = trans_matrix.shape[0]
    short_names = [n.split("/")[0].strip().split(" ")[0] + " " + n.split(" ")[-1]
                   for n in REGIME_NAMES[:n_states]]

    text_vals = [[f"{trans_matrix[i, j]:.2f}" for j in range(n_states)] for i in range(n_states)]

    fig = go.Figure(go.Heatmap(
        z=trans_matrix,
        x=short_names,
        y=short_names,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=9, family="Roboto Mono"),
        colorscale=[
            [0.0,  "rgba(5, 10, 18, 1)"],
            [0.3,  "rgba(0, 40, 80, 1)"],
            [0.6,  "rgba(0, 100, 50, 1)"],
            [0.85, "rgba(0, 200, 100, 1)"],
            [1.0,  "rgba(0, 255, 136, 1)"],
        ],
        reversescale=False,
        zmin=0, zmax=1,
        hovertemplate=(
            "From: <b>%{y}</b><br>"
            "To: <b>%{x}</b><br>"
            "Probability: <b>%{z:.3f}</b><extra></extra>"
        ),
        colorbar=dict(
            tickfont=dict(family="Roboto Mono", size=9, color="#8baac9"),
            thickness=12,
            len=0.85,
        ),
    ))

    fig.update_layout(
        height=height,
        title="Regime Transition Matrix (HMM)",
        xaxis=dict(tickfont=dict(family="Roboto Mono", size=9, color="#8baac9"), side="bottom"),
        yaxis=dict(tickfont=dict(family="Roboto Mono", size=9, color="#8baac9"), autorange="reversed"),
        **{k: v for k, v in _DARK_LAYOUT.items() if k not in ("xaxis", "yaxis", "hovermode")},
    )
    return fig


# ── Directional Probability Gauge ─────────────────────────────────────────

def plot_directional_gauge(p_up: float, p_down: float, confidence: float, height: int = 280) -> go.Figure:
    """Gauge chart showing directional bias (up vs down)."""
    # Gauge value: 0 = max bear, 50 = neutral, 100 = max bull
    gauge_val = p_up * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_val,
        delta={"reference": 50, "valueformat": ".1f", "suffix": "%"},
        title=dict(
            text=f"Directional Bias<br><span style='font-size:11px;color:#8baac9;font-family:Roboto Mono'>"
                 f"Confidence: {confidence:.1f}%</span>",
            font=dict(family="Roboto Mono", color="#e8f4fd", size=13),
        ),
        number=dict(suffix="%", font=dict(family="Roboto Mono", color="#00ff88", size=28)),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["🐻 100%", "75%", "↔ 50%", "75%", "🚀 100%"],
                tickfont=dict(family="Roboto Mono", size=8, color="#8baac9"),
            ),
            bar=dict(color="#00ff88", thickness=0.25),
            bgcolor="rgba(5, 10, 18, 0.8)",
            borderwidth=1,
            bordercolor="rgba(0, 170, 255, 0.2)",
            steps=[
                dict(range=[0, 35], color="rgba(255, 51, 102, 0.2)"),
                dict(range=[35, 65], color="rgba(170, 170, 200, 0.1)"),
                dict(range=[65, 100], color="rgba(0, 255, 136, 0.2)"),
            ],
            threshold=dict(
                line=dict(color="rgba(255, 170, 0, 0.8)", width=3),
                thickness=0.75,
                value=50,
            ),
        ),
    ))

    # Annotation: P(down)
    fig.add_annotation(
        x=0.15, y=0.15,
        text=f"P(↓) {p_down*100:.1f}%",
        showarrow=False,
        font=dict(family="Roboto Mono", size=11, color="#ff3366"),
        xref="paper", yref="paper",
    )
    # Annotation: P(up)
    fig.add_annotation(
        x=0.85, y=0.15,
        text=f"P(↑) {p_up*100:.1f}%",
        showarrow=False,
        font=dict(family="Roboto Mono", size=11, color="#00ff88"),
        xref="paper", yref="paper",
    )

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(5, 10, 18, 0.0)",
        font=dict(family="Roboto Mono", color="#8baac9"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ── Regime Timeline ────────────────────────────────────────────────────────

def plot_regime_timeline(tagged_df: pd.DataFrame, height: int = 100) -> go.Figure:
    """Color-coded regime timeline bar (horizontal stripe)."""
    fig = go.Figure()
    if "regime" not in tagged_df.columns:
        return fig

    regimes = tagged_df["regime"].values
    n = len(regimes)
    x = list(range(n))

    fig.add_trace(go.Scatter(
        x=x,
        y=[0.5] * n,
        mode="markers",
        marker=dict(
            color=[REGIME_COLORS[int(r)] for r in regimes],
            size=8,
            symbol="square",
            opacity=0.9,
        ),
        customdata=[[REGIME_NAMES[int(r)]] for r in regimes],
        hovertemplate="%{customdata[0]}<extra></extra>",
        name="Regime",
    ))

    fig.update_layout(
        height=height,
        showlegend=False,
        title="Regime Timeline",
        xaxis=dict(showticklabels=False, gridcolor="rgba(0,0,0,0)", showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, range=[0, 1]),
        margin=dict(l=50, r=20, t=40, b=10),
        paper_bgcolor="rgba(5, 10, 18, 0.0)",
        plot_bgcolor="rgba(5, 10, 18, 0.0)",
    )
    return fig


# ── Volatility Chart ───────────────────────────────────────────────────────

def plot_volatility_chart(tagged_df: pd.DataFrame, feature_df: pd.DataFrame = None, height: int = 280) -> go.Figure:
    """Realized volatility overlaid with ATR."""
    fig = go.Figure()

    log_ret = np.log(tagged_df["close"] / tagged_df["close"].shift(1))
    vol20 = log_ret.rolling(20).std() * np.sqrt(252) * 100

    fig.add_trace(go.Scatter(
        x=tagged_df.index,
        y=vol20,
        name="Realized Vol (20-bar)",
        line=dict(color="#ffaa00", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 170, 0, 0.07)",
    ))

    # ATR normalized
    h, l, c = tagged_df["high"], tagged_df["low"], tagged_df["close"]
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean() / tagged_df["close"] * 100

    fig.add_trace(go.Scatter(
        x=tagged_df.index,
        y=atr,
        name="ATR% (14)",
        line=dict(color="#aa66ff", width=1.2, dash="dot"),
    ))

    fig = _apply_dark_layout(fig, "Volatility — Realized Vol & ATR", height)
    fig.update_yaxes(ticksuffix="%")
    return fig


# ── Equity Curve ───────────────────────────────────────────────────────────

def plot_equity_curve(result: dict, height: int = 340) -> go.Figure:
    """Backtesting equity curve with drawdown shading."""
    equity = result.get("equity", pd.Series(dtype=float))
    drawdown = result.get("drawdown", pd.Series(dtype=float))

    if len(equity) == 0:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Equity Curve", "Drawdown"],
    )

    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        name="Strategy Equity",
        line=dict(color="#00ff88", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 255, 136, 0.05)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        name="Drawdown",
        line=dict(color="#ff3366", width=1.2),
        fill="tozeroy",
        fillcolor="rgba(255, 51, 102, 0.15)",
    ), row=2, col=1)

    fig.update_layout(
        height=height,
        showlegend=True,
        **{k: v for k, v in _DARK_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    fig.update_yaxes(row=1, col=1, tickprefix="$", tickfont=dict(size=9))
    fig.update_yaxes(row=2, col=1, ticksuffix="%", tickfont=dict(size=9))
    return fig


# ── Scenario Comparison Bar ────────────────────────────────────────────────

def plot_scenario_comparison(baseline: np.ndarray, shocked: np.ndarray, height: int = 340) -> go.Figure:
    """Side-by-side regime probability comparison for scenario simulation."""
    n_states = len(baseline)
    names = [f"{REGIME_ICONS[i]} {n.split('/')[0].strip()}" for i, n in enumerate(REGIME_NAMES[:n_states])]
    x = list(range(n_states))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline",
        x=names,
        y=[p * 100 for p in baseline],
        marker_color="rgba(0, 170, 255, 0.55)",
        marker_line_color="rgba(0, 170, 255, 0.8)",
        marker_line_width=1,
    ))
    fig.add_trace(go.Bar(
        name="Shocked",
        x=names,
        y=[p * 100 for p in shocked],
        marker_color=[
            "rgba(0, 255, 136, 0.7)" if s > b else "rgba(255, 51, 102, 0.7)"
            for b, s in zip(baseline, shocked)
        ],
        marker_line_color=[
            "#00ff88" if s > b else "#ff3366"
            for b, s in zip(baseline, shocked)
        ],
        marker_line_width=1,
    ))

    fig.update_layout(
        barmode="group",
        height=height,
        title="Regime Probabilities: Baseline vs Scenario",
        yaxis_ticksuffix="%",
        bargap=0.15,
        bargroupgap=0.05,
        **{k: v for k, v in _DARK_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    fig.update_xaxes(tickfont=dict(family="Roboto Mono", size=9, color="#8baac9"))
    fig.update_yaxes(tickfont=dict(size=9), gridcolor="rgba(0,170,255,0.08)")
    return fig


# ── Helper ────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"
