"""
Custom CSS for the hedge-fund terminal aesthetic.
Deep navy background, neon green/amber accents, monospace fonts, glassmorphism cards.
"""

DARK_THEME_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

/* ── Root Variables ───────────────────────────────────────────── */
:root {
    --bg-primary:    #050a12;
    --bg-secondary:  #0a1628;
    --bg-card:       rgba(10, 22, 40, 0.85);
    --bg-card-hover: rgba(15, 30, 55, 0.95);
    --accent-green:  #00ff88;
    --accent-amber:  #ffaa00;
    --accent-red:    #ff3366;
    --accent-blue:   #00aaff;
    --accent-purple: #aa66ff;
    --text-primary:  #e8f4fd;
    --text-secondary:#8baac9;
    --text-muted:    #4a6b8a;
    --border-color:  rgba(0, 170, 255, 0.15);
    --border-glow:   rgba(0, 255, 136, 0.4);
    --font-mono:     'Roboto Mono', 'Share Tech Mono', monospace;
    --font-ui:       'Inter', sans-serif;
    --radius:        8px;
    --shadow-glow:   0 0 20px rgba(0, 255, 136, 0.1);
}

/* ── Global App ────────────────────────────────────────────────── */
.stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(0, 100, 255, 0.05) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, rgba(0, 255, 136, 0.03) 0%, transparent 60%) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}

/* ── Sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-green) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* ── Metric KPI Cards ──────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    padding: 14px 18px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: var(--shadow-glow) !important;
    transition: all 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 25px rgba(0, 255, 136, 0.15) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-green) !important;
    font-family: var(--font-mono) !important;
    font-size: 1.5rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
}

/* ── Tabs ──────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-green) !important;
    border-bottom-color: var(--accent-green) !important;
}

/* ── Selectboxes & Sliders ─────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}
.stSlider > div {
    color: var(--text-primary) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-green) !important;
}

/* ── Buttons ───────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.15), rgba(0, 170, 255, 0.1)) !important;
    border: 1px solid var(--accent-green) !important;
    color: var(--accent-green) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    border-radius: var(--radius) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.3), rgba(0, 170, 255, 0.2)) !important;
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Progress Bars ─────────────────────────────────────────────── */
.stProgress > div > div {
    background-color: var(--accent-green) !important;
}

/* ── Custom Card ───────────────────────────────────────────────── */
.regime-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 16px 20px;
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow-glow);
    margin-bottom: 12px;
}
.regime-card:hover {
    border-color: var(--accent-green);
    box-shadow: 0 0 25px rgba(0, 255, 136, 0.15);
}

/* ── Terminal Header ───────────────────────────────────────────── */
.terminal-header {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-color);
}

/* ── Badge ─────────────────────────────────────────────────────── */
.regime-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    border: 1px solid;
}

/* ── Confidence Meter ──────────────────────────────────────────── */
.confidence-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--text-muted), var(--accent-green));
    margin-top: 6px;
    transition: width 0.5s ease;
}

/* ── Divider ───────────────────────────────────────────────────── */
hr {
    border-color: var(--border-color) !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-green); }

/* ── DataFrames ────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
}
</style>
"""


def inject_css():
    """Call this at the top of every Streamlit page."""
    import streamlit as st
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
