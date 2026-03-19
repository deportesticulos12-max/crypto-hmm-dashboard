"""
Directional Probability Engine.

Estimates P(up), P(down), P(flat) using:
  1. Current HMM regime posterior probabilities
  2. Historically observed conditional return distributions per regime
  3. Bayesian update with recent 5-bar realized returns
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from config.settings import REGIME_NAMES, REGIME_DIRECTIONAL_BIAS


@dataclass
class DirectionalForecast:
    p_up: float           # probability price moves up next bar
    p_down: float         # probability price moves down next bar
    p_flat: float         # probability price stays rangebound
    confidence: float     # 0-100 overall confidence score
    p_high_vol: float     # probability of high volatility regime
    p_transition: float   # probability of regime transition next bar
    regime_proba: np.ndarray  # latest bar's 7-regime probabilities
    current_regime: int   # argmax regime
    current_regime_name: str


class DirectionalEngine:
    """
    Computes directional forecasts by combining HMM regime posteriors
    with regime-conditional return statistics.
    """

    def __init__(self, n_states: int = 7):
        self.n_states = n_states
        # Regime-conditional return distributions (fit from data)
        self._regime_mean_ret: np.ndarray = np.zeros(n_states)
        self._regime_std_ret: np.ndarray = np.ones(n_states) * 0.01
        
        # Base probabilities are padded/truncated to match the number of states
        base_up = [0.65, 0.60, 0.50, 0.50, 0.40, 0.35, 0.25]
        base_down = [0.20, 0.25, 0.30, 0.35, 0.45, 0.50, 0.65]
        
        if n_states > len(base_up):
            base_up += [0.50] * (n_states - len(base_up))
            base_down += [0.50] * (n_states - len(base_down))
            
        self._regime_p_up: np.ndarray = np.array(base_up)[:n_states]
        self._regime_p_down: np.ndarray = np.array(base_down)[:n_states]
        
        self._fitted = False

    def fit(self, tagged_df: pd.DataFrame):
        """
        Learn regime-conditional return distributions from historical data.
        """
        log_ret = np.log(tagged_df["close"] / tagged_df["close"].shift(1)).fillna(0)
        tagged_df = tagged_df.copy()
        tagged_df["log_return"] = log_ret

        for i in range(self.n_states):
            mask = tagged_df["regime"] == i
            subset = tagged_df[mask]["log_return"]
            if len(subset) > 5:
                self._regime_mean_ret[i] = float(subset.mean())
                self._regime_std_ret[i] = float(subset.std()) + 1e-8
                self._regime_p_up[i] = float((subset > 0).mean())
                self._regime_p_down[i] = float((subset < 0).mean())

        self._fitted = True

    def forecast(
        self,
        regime_proba: np.ndarray,
        transition_matrix: np.ndarray,
        recent_returns: np.ndarray,
    ) -> DirectionalForecast:
        """
        Compute directional forecast from latest regime probabilities.

        Parameters
        ----------
        regime_proba : (n_states,) posterior probabilities for latest bar
        transition_matrix : (n_states, n_states) HMM A matrix
        recent_returns : array of last 5 log-returns (for Bayesian update)
        """
        # ── Step 1: Weighted P(up) / P(down) across regimes ─────────────────
        p_up_base = float(np.dot(regime_proba, self._regime_p_up))
        p_down_base = float(np.dot(regime_proba, self._regime_p_down))
        p_flat_base = max(0.0, 1.0 - p_up_base - p_down_base)

        # ── Step 2: Bayesian update from recent returns ──────────────────────
        if len(recent_returns) > 0:
            recent_bias = float(np.mean(recent_returns))
            # Shift probabilities slightly toward direction of recent momentum
            momentum_strength = np.tanh(recent_bias * 50)  # scale: -1 to +1
            update_factor = 0.15  # how much to nudge
            if momentum_strength > 0:
                p_up_adj = p_up_base + update_factor * momentum_strength
                p_down_adj = p_down_base - update_factor * momentum_strength * 0.5
            else:
                p_up_adj = p_up_base + update_factor * momentum_strength * 0.5
                p_down_adj = p_down_base - update_factor * momentum_strength
            # Clamp and renormalize
            p_up_adj = max(0.05, min(0.90, p_up_adj))
            p_down_adj = max(0.05, min(0.90, p_down_adj))
            total = p_up_adj + p_down_adj + p_flat_base
            p_up_final = p_up_adj / total
            p_down_final = p_down_adj / total
            p_flat_final = p_flat_base / total
        else:
            p_up_final = p_up_base
            p_down_final = p_down_base
            p_flat_final = p_flat_base

        # ── Step 3: Confidence score 0–100 ──────────────────────────────────
        # High confidence = high probability concentrated on one regime
        dominant_regime_prob = float(np.max(regime_proba))
        # Signal strength: max of up/down probability
        signal_strength = max(p_up_final, p_down_final) - 0.5
        signal_strength = max(0.0, signal_strength) * 2   # 0–1 range
        # Blend regime concentration + signal strength
        confidence = (dominant_regime_prob * 0.6 + signal_strength * 0.4) * 100
        confidence = min(100.0, max(0.0, confidence))

        # ── Step 4: High vol probability ─────────────────────────────────────
        # State 3 = Volatility Expansion (semantic index 3)
        high_vol_regimes = [3]  # "Volatility Expansion"
        p_high_vol = float(sum(regime_proba[i] for i in high_vol_regimes))

        # ── Step 5: Regime transition probability ────────────────────────────
        current_regime = int(np.argmax(regime_proba))
        # P(transition) = 1 - P(stay in same regime)
        p_stay = float(transition_matrix[current_regime, current_regime])
        p_transition = 1.0 - p_stay

        current_regime_name = REGIME_NAMES[current_regime]

        return DirectionalForecast(
            p_up=round(p_up_final, 4),
            p_down=round(p_down_final, 4),
            p_flat=round(p_flat_final, 4),
            confidence=round(confidence, 2),
            p_high_vol=round(p_high_vol, 4),
            p_transition=round(p_transition, 4),
            regime_proba=regime_proba,
            current_regime=current_regime,
            current_regime_name=current_regime_name,
        )

    def bulk_forecast(
        self,
        all_proba: np.ndarray,
        transition_matrix: np.ndarray,
        log_returns: np.ndarray,
        lookback: int = 5,
    ) -> pd.DataFrame:
        """
        Run forecast for all bars, returning a DataFrame with columns:
        p_up, p_down, p_flat, confidence, regime.
        """
        rows = []
        n = len(all_proba)
        for t in range(n):
            recent = log_returns[max(0, t - lookback):t]
            fc = self.forecast(all_proba[t], transition_matrix, recent)
            rows.append({
                "p_up": fc.p_up,
                "p_down": fc.p_down,
                "p_flat": fc.p_flat,
                "confidence": fc.confidence,
                "p_high_vol": fc.p_high_vol,
                "p_transition": fc.p_transition,
            })
        return pd.DataFrame(rows)
