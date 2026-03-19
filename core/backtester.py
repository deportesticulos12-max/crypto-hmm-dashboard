"""
Regime-based backtesting engine with multiple strategies.
"""

import numpy as np
import pandas as pd
from utils.metrics import compute_all_metrics
from config.settings import (
    DEFAULT_INITIAL_CAPITAL, DEFAULT_POSITION_SIZE, TRADING_FEE,
    BULL_THRESHOLD, BEAR_THRESHOLD, NEUTRAL_THRESHOLD, REGIME_NAMES
)


STRATEGY_DESCRIPTIONS = {
    "trend_following": "Long in bull regimes (0,1), flat otherwise",
    "bear_short": "Short in bear regimes (5,6), flat otherwise",
    "neutral_hedge": "Flat in sideways/vol regimes (2,3), long/short elsewhere",
    "combined": "Trend-following + bear-short combined strategy",
    "regime_rotation": "Full rotation: long bulls, short bears, flat sideways/vol",
}


class RegimeBacktester:
    """
    Backtests regime-based trading strategies on historical data.
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        position_size: float = DEFAULT_POSITION_SIZE,
        fee: float = TRADING_FEE,
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.fee = fee

    def _generate_signals(
        self,
        tagged_df: pd.DataFrame,
        strategy: str,
        prob_threshold: float,
    ) -> pd.Series:
        """
        Generate position signals: +1 (long), -1 (short), 0 (flat).
        """
        regime = tagged_df["regime"]
        # Get max regime probability per bar
        prob_cols = [c for c in tagged_df.columns if c.startswith("regime_prob_")]
        if prob_cols:
            max_prob = tagged_df[prob_cols].max(axis=1)
        else:
            max_prob = pd.Series(1.0, index=tagged_df.index)

        signal = pd.Series(0, index=tagged_df.index, dtype=float)

        if strategy == "trend_following":
            bull_mask = (regime.isin([0, 1])) & (max_prob >= prob_threshold)
            signal[bull_mask] = 1

        elif strategy == "bear_short":
            bear_mask = (regime.isin([5, 6])) & (max_prob >= prob_threshold)
            signal[bear_mask] = -1

        elif strategy == "neutral_hedge":
            bull_mask = (regime.isin([0, 1])) & (max_prob >= prob_threshold)
            bear_mask = (regime.isin([5, 6])) & (max_prob >= prob_threshold)
            signal[bull_mask] = 1
            signal[bear_mask] = -1

        elif strategy == "combined":
            bull_mask = (regime.isin([0, 1])) & (max_prob >= prob_threshold)
            bear_mask = (regime.isin([5, 6])) & (max_prob >= prob_threshold)
            signal[bull_mask] = 1
            signal[bear_mask] = -1

        elif strategy == "regime_rotation":
            # Long: strong bull, bull
            signal[regime.isin([0, 1]) & (max_prob >= prob_threshold)] = 1
            # Flat: accumulation, vol expansion, distribution
            signal[regime.isin([2, 3, 4])] = 0
            # Short: bear, strong bear
            signal[regime.isin([5, 6]) & (max_prob >= prob_threshold)] = -1

        return signal

    def run(
        self,
        tagged_df: pd.DataFrame,
        strategy: str = "trend_following",
        prob_threshold: float = BULL_THRESHOLD,
        use_close_returns: bool = True,
    ) -> dict:
        """
        Run backtest and return dict with equity curve and performance metrics.
        """
        if len(tagged_df) < 10:
            return _empty_result()

        df = tagged_df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        signals = self._generate_signals(df, strategy, prob_threshold)

        # Shift signals by 1 bar (avoid look-ahead bias)
        positions = signals.shift(1).fillna(0)

        # PnL calculation
        gross_returns = positions * df["log_return"]

        # Apply trading fee on position changes
        position_changes = positions.diff().abs().fillna(0)
        fee_cost = position_changes * self.fee
        net_returns = gross_returns - fee_cost

        # Scale by position size
        strategy_returns = net_returns * self.position_size

        # Equity curve
        equity = pd.Series(
            self.initial_capital * (1 + strategy_returns).cumprod(),
            index=df.index
        )

        # Drawdown series
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max

        # Regime-level P&L
        regime_pnl = {}
        n_states = int(df["regime"].max() + 1) if not df["regime"].empty else len(REGIME_NAMES)
        for i in range(n_states):
            name = REGIME_NAMES[i]
            mask = df["regime"] == i
            regime_returns = strategy_returns[mask]
            regime_pnl[name] = {
                "total_return": float(regime_returns.sum()),
                "n_bars": int(mask.sum()),
                "win_rate": float((regime_returns > 0).mean()) if len(regime_returns) > 0 else 0.0,
            }

        # Determine periods_per_year from average trade frequency
        metrics = compute_all_metrics(strategy_returns, equity, periods_per_year=252)

        return {
            "strategy": strategy,
            "equity": equity,
            "returns": strategy_returns,
            "drawdown": drawdown,
            "signals": signals,
            "positions": positions,
            "metrics": metrics,
            "regime_pnl": regime_pnl,
            "n_trades": int((positions.diff().abs() > 0).sum()),
        }

    def run_all_strategies(
        self,
        tagged_df: pd.DataFrame,
        prob_threshold: float = BULL_THRESHOLD,
    ) -> dict:
        """Run all strategies and return a comparison dict."""
        results = {}
        for strat in STRATEGY_DESCRIPTIONS:
            try:
                results[strat] = self.run(tagged_df, strat, prob_threshold)
            except Exception as e:
                results[strat] = _empty_result()
        return results


def _empty_result() -> dict:
    return {
        "strategy": "N/A",
        "equity": pd.Series(dtype=float),
        "returns": pd.Series(dtype=float),
        "drawdown": pd.Series(dtype=float),
        "signals": pd.Series(dtype=float),
        "positions": pd.Series(dtype=float),
        "metrics": {
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "cagr": 0.0, "total_trades": 0, "net_return": 0.0,
        },
        "regime_pnl": {},
        "n_trades": 0,
    }
