"""
Regime analyzer: statistics, persistence, duration, and DataFrame tagging.
"""

import numpy as np
import pandas as pd
from typing import Optional
from config.settings import REGIME_NAMES, REGIME_COLORS, REGIME_ICONS


class RegimeAnalyzer:
    """
    Computes per-regime statistics from decoded regime sequences and OHLCV data.
    """

    def __init__(self, n_states: int = 7):
        self.n_states = n_states

    def tag_dataframe(
        self,
        df: pd.DataFrame,
        regimes: np.ndarray,
        proba: np.ndarray,
    ) -> pd.DataFrame:
        """
        Append regime labels and probabilities to the OHLCV DataFrame.
        Note: feature engineering drops some leading bars (NaN warm-up),
        so we align from the end of df.
        """
        n = len(regimes)
        tagged = df.copy()
        # Align: features are computed for the last `n` rows of df
        tagged = tagged.iloc[-n:].copy()
        tagged["regime"] = regimes
        tagged["regime_name"] = [REGIME_NAMES[r] for r in regimes]
        tagged["regime_color"] = [REGIME_COLORS[r] for r in regimes]
        tagged["regime_icon"] = [REGIME_ICONS[r] for r in regimes]

        for i in range(self.n_states):
            col = f"prob_{REGIME_NAMES[i].split('/')[0].strip().replace(' ', '_').lower()}"
            tagged[f"regime_prob_{i}"] = proba[:, i]

        return tagged

    def compute_regime_stats(
        self,
        tagged_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute per-regime statistics: mean return, vol, frequency, avg duration.
        Returns a DataFrame indexed by regime index.
        """
        log_ret = np.log(tagged_df["close"] / tagged_df["close"].shift(1)).fillna(0)
        tagged_df = tagged_df.copy()
        tagged_df["log_return"] = log_ret

        rows = []
        for i in range(self.n_states):
            mask = tagged_df["regime"] == i
            subset = tagged_df[mask]
            ret = subset["log_return"]
            count = len(subset)
            freq = count / len(tagged_df) if len(tagged_df) > 0 else 0

            # Average consecutive duration
            durations = self._compute_durations(tagged_df["regime"].values, i)
            avg_dur = np.mean(durations) if durations else 0.0

            rows.append({
                "regime": i,
                "name": REGIME_NAMES[i],
                "icon": REGIME_ICONS[i],
                "color": REGIME_COLORS[i],
                "mean_return": float(ret.mean()) if count > 0 else 0.0,
                "volatility": float(ret.std() * np.sqrt(252)) if count > 1 else 0.0,
                "frequency": float(freq),
                "avg_duration_bars": float(avg_dur),
                "count": int(count),
            })

        return pd.DataFrame(rows).set_index("regime")

    def _compute_durations(self, regime_seq: np.ndarray, regime_id: int) -> list:
        """Compute list of consecutive run lengths for a given regime."""
        durations = []
        current_len = 0
        for r in regime_seq:
            if r == regime_id:
                current_len += 1
            else:
                if current_len > 0:
                    durations.append(current_len)
                    current_len = 0
        if current_len > 0:
            durations.append(current_len)
        return durations

    def get_current_regime_bar_count(self, tagged_df: pd.DataFrame) -> int:
        """How many bars has the market been in the current regime consecutively?"""
        last_regime = tagged_df["regime"].iloc[-1]
        count = 0
        for r in reversed(tagged_df["regime"].values):
            if r == last_regime:
                count += 1
            else:
                break
        return count

    def compute_regime_transitions(self, tagged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute empirical transition frequency matrix from decoded regime sequence.
        Returns DataFrame of shape (n_states, n_states).
        """
        seq = tagged_df["regime"].values
        counts = np.zeros((self.n_states, self.n_states))
        for t in range(len(seq) - 1):
            i, j = seq[t], seq[t + 1]
            counts[i, j] += 1
        # Normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        trans = counts / row_sums
        return pd.DataFrame(
            trans,
            index=REGIME_NAMES[:self.n_states],
            columns=REGIME_NAMES[:self.n_states],
        )

    def expected_regime_duration(self, transition_matrix: np.ndarray) -> np.ndarray:
        """
        Expected duration of each regime from HMM transition matrix diagonal.
        duration_i = 1 / (1 - A[i,i])
        """
        diag = np.diag(transition_matrix)
        diag = np.clip(diag, 0, 1 - 1e-6)
        return 1.0 / (1.0 - diag + 1e-10)
