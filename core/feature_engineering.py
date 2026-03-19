"""
Feature engineering: compute a rich set of price/vol/momentum/volume/crypto features
and normalize them for HMM input.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config.settings import (
    MA_WINDOWS, RSI_PERIOD, ATR_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    STOCH_K, STOCH_D, ROLLING_VOL_WINDOWS, MOMENTUM_PERIOD, CHOPPINESS_PERIOD
)


def _safe_div(a, b, fill=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result


class FeatureEngineer:
    """
    Transforms a raw OHLCV DataFrame into a normalized feature matrix
    suitable for HMM training.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: list = []
        self._fitted = False

    # ── Price Features ────────────────────────────────────────────────────────

    def _compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        c = df["close"]
        out["log_return"] = np.log(c / c.shift(1))
        for w in [5, 10, 20, 50]:
            out[f"roll_ret_{w}"] = c.pct_change(w)
        # Trend slope: linear regression slope over 20 bars (normalized)
        out["trend_slope"] = (
            c.rolling(20)
            .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean(), raw=True)
        )
        # Price position relative to own range (0–1)
        rolling_low = df["low"].rolling(20).min()
        rolling_high = df["high"].rolling(20).max()
        out["price_position"] = _safe_div(c - rolling_low, rolling_high - rolling_low)
        return out

    # ── Volatility Features ───────────────────────────────────────────────────

    def _compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        for w in ROLLING_VOL_WINDOWS:
            out[f"realized_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)
        # ATR
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(ATR_PERIOD).mean()
        out["atr_norm"] = _safe_div(atr.values, c.values)
        # Volatility regime: current vol vs long-run vol (vol-of-vol signal)
        vol5 = log_ret.rolling(5).std()
        vol20 = log_ret.rolling(20).std()
        out["vol_ratio"] = _safe_div(vol5.values, vol20.values + 1e-10)
        # ── Garman-Klass Volatility ──
        # Formula: 0.5 * [ln(H/L)]^2 - (2ln(2)-1) * [ln(C/O)]^2
        # where O, H, L, C are Open, High, Low, Close
        if "open" in df.columns:
            log_hl = np.log(_safe_div(h.values, l.values, fill=1.0))
            log_co = np.log(_safe_div(c.values, df["open"].values, fill=1.0))
            rs = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)
            # Clip negative values that might occur due to numerical precision issues
            rs = np.clip(rs, a_min=0.0, a_max=None)
            for w in ROLLING_VOL_WINDOWS:
                out[f"gk_vol_{w}"] = np.sqrt(pd.Series(rs).rolling(w).mean()) * np.sqrt(252)
        else:
            # Fallback to standard realized vol if open isn't available
            for w in ROLLING_VOL_WINDOWS:
                out[f"gk_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

        return out

    # ── Momentum Features ─────────────────────────────────────────────────────

    def _compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        c = df["close"]

        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        rs = _safe_div(gain.values, loss.values + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        out["rsi"] = (rsi - 50) / 50   # normalized to [-1, 1]

        # MACD
        ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        macd_hist = macd_line - signal_line
        out["macd_hist"] = macd_hist / c.rolling(50).std().replace(0, np.nan)

        # Stochastic
        low_min = df["low"].rolling(STOCH_K).min()
        high_max = df["high"].rolling(STOCH_K).max()
        stoch_k = 100 * _safe_div(
            (c - low_min).values, (high_max - low_min).values + 1e-10
        )
        out["stoch_k"] = (pd.Series(stoch_k, index=df.index).rolling(STOCH_D).mean() - 50) / 50

        # Raw momentum
        out["momentum"] = _safe_div(
            (c - c.shift(MOMENTUM_PERIOD)).values, c.shift(MOMENTUM_PERIOD).values + 1e-10
        )

        # Rate of change
        out["roc_5"] = c.pct_change(5)
        out["roc_20"] = c.pct_change(20)
        return out

    # ── Trend Features ────────────────────────────────────────────────────────

    def _compute_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        c = df["close"]
        for w in MA_WINDOWS:
            ma = c.rolling(w).mean()
            out[f"dist_ma{w}"] = _safe_div((c - ma).values, ma.values + 1e-10)

        # EMA spreads
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        ema200 = c.ewm(span=200, adjust=False).mean()
        out["ema20_50_spread"] = _safe_div((ema20 - ema50).values, ema50.values + 1e-10)
        out["ema50_200_spread"] = _safe_div((ema50 - ema200).values, ema200.values + 1e-10)

        # Market structure: distance from 20-bar high/low
        roll_high = df["high"].rolling(20).max()
        roll_low = df["low"].rolling(20).min()
        out["dist_roll_high"] = _safe_div((c - roll_high).values, roll_high.values + 1e-10)
        out["dist_roll_low"] = _safe_div((c - roll_low).values, roll_low.values + 1e-10)

        # ── Choppiness Index (Trend vs Ranging Proxy) ──
        # 100 * LOG10( SUM(ATR(1), n) / (MaxHigh(n) - MinLow(n)) ) / LOG10(n)
        # Closer to 100 = Choppy (Ranging), Closer to 0 = Trending
        n = CHOPPINESS_PERIOD if "CHOPPINESS_PERIOD" in globals() else 14
        h = df["high"]
        l = df["low"]
        tr1 = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)

        sum_tr = tr1.rolling(n).sum()
        max_h = h.rolling(n).max()
        min_l = l.rolling(n).min()

        range_hl = max_h - min_l
        chop = 100 * np.log10(_safe_div(sum_tr.values, range_hl.values + 1e-10)) / np.log10(n)
        # Normalize roughly around 50
        out["choppiness"] = (chop - 50) / 25.0

        return out

    # ── Volume Features ───────────────────────────────────────────────────────

    def _compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        vol = df["volume"]
        vol_ma = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std()
        out["volume_zscore"] = _safe_div((vol - vol_ma).values, vol_std.values + 1e-10)
        out["volume_ma_ratio"] = _safe_div(vol.values, vol_ma.values + 1e-10)

        # Taker-buy ratio — proxy for buying vs selling pressure
        if "taker_buy_base" in df.columns:
            out["taker_ratio"] = _safe_div(
                df["taker_buy_base"].values, vol.values + 1e-10
            )
        else:
            out["taker_ratio"] = 0.5

        # Quote volume (dollar volume) z-score
        if "quote_asset_volume" in df.columns:
            qvol = df["quote_asset_volume"]
            qvol_ma = qvol.rolling(20).mean()
            qvol_std = qvol.rolling(20).std()
            out["qvol_zscore"] = _safe_div((qvol - qvol_ma).values, qvol_std.values + 1e-10)
        return out

    # ── Crypto-Specific Features ──────────────────────────────────────────────

    def _compute_crypto_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        if "funding_rate" in df.columns:
            out["funding_rate"] = df["funding_rate"].fillna(0.0)
            out["funding_sign"] = np.sign(df["funding_rate"].fillna(0.0))
        else:
            out["funding_rate"] = 0.0
            out["funding_sign"] = 0.0

        if "oi_change" in df.columns:
            out["oi_change"] = df["oi_change"].fillna(0.0)
        else:
            out["oi_change"] = 0.0

        return out

    # ── Normalize ─────────────────────────────────────────────────────────────

    def _normalize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            X = self.scaler.fit_transform(X)
            self._fitted = True
        else:
            X = self.scaler.transform(X)
        return X

    # ── Public API ────────────────────────────────────────────────────────────

    def transform(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute all features and return (X_array, feature_names).
        X_array shape: (n_bars, n_features), aligned and normalized.
        """
        feature_frames = [
            self._compute_price_features(df),
            self._compute_volatility_features(df),
            self._compute_momentum_features(df),
            self._compute_trend_features(df),
            self._compute_volume_features(df),
            self._compute_crypto_features(df),
        ]
        combined = pd.concat(feature_frames, axis=1)
        combined = combined.replace([np.inf, -np.inf], np.nan)

        # Drop rows where >50% features are NaN (warm-up period)
        threshold = int(len(combined.columns) * 0.5)
        combined = combined.dropna(thresh=threshold)
        combined = combined.fillna(0.0)

        self.feature_names = list(combined.columns)
        X = self._normalize(combined.values, fit=fit_scaler)
        return X, self.feature_names

    def transform_live(self, df: pd.DataFrame) -> np.ndarray:
        """Transform without fitting (use existing scaler)."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted yet — call transform() first.")
        X, _ = self.transform(df, fit_scaler=False)
        return X

    def get_feature_df(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Return features as a DataFrame (for inspection/visualization)."""
        X, names = self.transform(df, fit_scaler=fit_scaler)
        # Reconstruct index — feature df may be shorter due to NaN drop
        idx_len = min(len(df), X.shape[0])
        idx = df.index[-idx_len:]
        return pd.DataFrame(X, index=idx, columns=names)
