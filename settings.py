"""
Global settings and constants for the Crypto HMM Regime Detection Dashboard.
"""

# ── API Endpoints ────────────────────────────────────────────────────────────
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# ── Supported Assets ─────────────────────────────────────────────────────────
ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "ADAUSDT", "MATICUSDT", "AVAXUSDT", "DOGEUSDT",
    "XRPUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
]

ASSET_DISPLAY = {s: s.replace("USDT", "") for s in ASSET_LIST}

# ── Supported Timeframes ─────────────────────────────────────────────────────
TIMEFRAME_LIST = ["1m", "5m", "15m", "1h", "4h", "1d"]
TIMEFRAME_BARS = {
    "1m": 500, "5m": 500, "15m": 500,
    "1h": 600, "4h": 700, "1d": 800,
}

# ── HMM Model Parameters ─────────────────────────────────────────────────────
HMM_N_STATES = 7
HMM_N_ITER = 300
HMM_COVARIANCE_TYPE = "full"
HMM_N_RESTARTS = 5          # multiple random restarts → pick best log-likelihood
HMM_RANDOM_STATE = 42

# ── 7 Market Regimes ─────────────────────────────────────────────────────────
REGIME_NAMES = [
    "Strong Bull Expansion",
    "Bull Continuation",
    "Accumulation / Sideways",
    "Volatility Expansion",
    "Distribution Phase",
    "Bear Continuation",
    "Strong Bear Capitulation",
]

# Color palette: neon-compatible for dark UI
REGIME_COLORS = [
    "#00FF88",   # Strong Bull — bright neon green
    "#44DD66",   # Bull Continuation — softer green
    "#AAAACC",   # Accumulation — muted purple-grey
    "#FFAA00",   # Volatility Expansion — amber
    "#FF6644",   # Distribution — orange-red
    "#FF3366",   # Bear Continuation — hot pink-red
    "#CC0033",   # Strong Bear Capitulation — deep red
]

REGIME_ICONS = ["🚀", "📈", "↔️", "⚡", "📦", "📉", "💀"]

# Regime directional bias: expected sign of returns per regime
# +1 = bullish, 0 = neutral, -1 = bearish
REGIME_DIRECTIONAL_BIAS = [1, 1, 0, 0, -1, -1, -1]

# ── Feature Engineering Parameters ──────────────────────────────────────────
MA_WINDOWS = [20, 50, 100, 200]
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOCH_K = 14
STOCH_D = 3
ROLLING_VOL_WINDOWS = [5, 10, 20]
MOMENTUM_PERIOD = 10
CHOPPINESS_PERIOD = 14

# ── Backtesting Parameters ───────────────────────────────────────────────────
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_POSITION_SIZE = 0.1    # fraction of capital per trade
TRADING_FEE = 0.001            # 0.1% per trade (Binance taker)

# Regime probability threshold to take a trade
BULL_THRESHOLD = 0.50
BEAR_THRESHOLD = 0.50
NEUTRAL_THRESHOLD = 0.40

# ── Dashboard UI ─────────────────────────────────────────────────────────────
APP_TITLE = "CryptoHMM — Regime Intelligence Terminal"
APP_ICON = "🧬"
REFRESH_INTERVAL_OPTIONS = [30, 60, 120, 300]  # seconds
DEFAULT_REFRESH = 60

# ── Cache Settings ───────────────────────────────────────────────────────────
CACHE_DIR = "models/.cache"
CACHE_TTL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 14400, "1d": 86400,
}

# ── Scenario Simulation ──────────────────────────────────────────────────────
SCENARIOS = {
    "Volatility Spike": {
        "description": "Sudden VIX-like shock — vol doubles, range expands",
        "icon": "⚡",
        "shocks": {"realized_vol": 3.0, "atr_norm": 2.5, "rsi": -15, "volume_zscore": 2.0},
    },
    "Liquidation Cascade": {
        "description": "Mass liquidations — sharp drop, vol spike, OI collapse",
        "icon": "💥",
        "shocks": {"log_return": -3.0, "realized_vol": 4.0, "volume_zscore": 3.0, "rsi": -25},
    },
    "Bull Breakout": {
        "description": "Strong breakout above resistance — momentum surge",
        "icon": "🚀",
        "shocks": {"log_return": 2.5, "rsi": 15, "momentum": 2.0, "volume_zscore": 1.5},
    },
    "Bear Breakdown": {
        "description": "Support break — trend reversal signal",
        "icon": "🐻",
        "shocks": {"log_return": -2.5, "rsi": -15, "momentum": -2.0, "macd_hist": -1.5},
    },
    "Volume Expansion": {
        "description": "Massive volume surge — smart money accumulation",
        "icon": "📊",
        "shocks": {"volume_zscore": 4.0, "volume_ma_ratio": 2.5, "realized_vol": 1.2},
    },
    "Liquidity Shock": {
        "description": "Flash crash / liquidity drain — spread widens",
        "icon": "🌊",
        "shocks": {"log_return": -4.0, "atr_norm": 3.0, "realized_vol": 5.0, "rsi": -30},
    },
    "Macro Crash": {
        "description": "Correlated sell-off — all risk assets fall together",
        "icon": "🔴",
        "shocks": {"log_return": -5.0, "realized_vol": 6.0, "rsi": -35, "macd_hist": -2.5, "volume_zscore": 2.5},
    },
}
