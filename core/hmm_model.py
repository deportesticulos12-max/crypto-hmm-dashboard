"""
7-state Hidden Markov Model for cryptocurrency market regime detection.
Uses hmmlearn GaussianHMM with multiple random restarts and Viterbi decoding.
"""

import numpy as np
import joblib
import logging
import os
from typing import Optional

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    # ── Fallback for Windows without C++ tools ──
    from sklearn.mixture import GaussianMixture
    class GaussianHMM:
        def __init__(self, n_components=1, covariance_type='diag', n_iter=100,
                     random_state=None, tol=1e-2, verbose=False):
            self.n_components = n_components
            self.model = GaussianMixture(
                n_components=n_components, covariance_type=covariance_type,
                max_iter=n_iter, random_state=random_state, tol=tol
            )
            self.transmat_ = None
            self.means_ = None

        def fit(self, X, y=None):
            self.model.fit(X)
            self.means_ = self.model.means_
            preds = self.model.predict(X)
            n = self.n_components
            tm = np.zeros((n, n))
            for i in range(len(preds) - 1):
                tm[preds[i], preds[i+1]] += 1
            tm += 1e-6
            self.transmat_ = tm / tm.sum(axis=1, keepdims=True)
            return self

        def score(self, X, y=None):
            return self.model.score(X) * len(X)

        def decode(self, X, algorithm="viterbi"):
            return 0.0, self.model.predict(X)

        def predict_proba(self, X):
            return self.model.predict_proba(X)

from sklearn.utils import check_random_state
from config.settings import (
    HMM_N_STATES, HMM_N_ITER, HMM_COVARIANCE_TYPE,
    HMM_N_RESTARTS, HMM_RANDOM_STATE, REGIME_NAMES, REGIME_DIRECTIONAL_BIAS
)

logger = logging.getLogger(__name__)


class CryptoHMM:
    """
    7-state Hidden Markov Model for market regime detection.

    Attributes
    ----------
    n_states : int
        Number of hidden states (default 7).
    model : GaussianHMM
        The fitted hmmlearn model.
    regime_map : list[int]
        Maps sorted-by-mean-return state index → original model state index.
    """

    def __init__(self, n_states: int = HMM_N_STATES):
        self.n_states = n_states
        self.model: Optional[GaussianHMM] = None
        self._log_likelihood = -np.inf
        self.regime_map: list = list(range(n_states))  # identity by default
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray) -> float:
        """
        Train the HMM using Baum-Welch EM with multiple random restarts.
        Returns best log-likelihood found.
        """
        best_ll = -np.inf
        best_model = None

        for seed in range(HMM_RANDOM_STATE, HMM_RANDOM_STATE + HMM_N_RESTARTS):
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=HMM_COVARIANCE_TYPE,
                    n_iter=HMM_N_ITER,
                    random_state=seed,
                    verbose=False,
                    tol=1e-4,
                )
                model.fit(X)
                ll = model.score(X)
                logger.info(f"Restart {seed - HMM_RANDOM_STATE + 1}: log-likelihood = {ll:.4f}")
                if ll > best_ll:
                    best_ll = ll
                    best_model = model
            except Exception as e:
                logger.warning(f"HMM training restart failed: {e}")
                continue

        if best_model is None:
            raise RuntimeError("All HMM training restarts failed.")

        self.model = best_model
        self._log_likelihood = best_ll
        self._compute_regime_map(X)
        self._trained = True
        return best_ll

    # ── Regime Semantic Sorting ───────────────────────────────────────────────

    def _compute_regime_map(self, X: np.ndarray):
        """
        Sort states by the first feature mean (assumed to be log_return).
        State with highest mean → "Strong Bull Expansion" (index 0).
        State with lowest mean → "Strong Bear Capitulation" (index 6).
        """
        # Use mean of first feature (log_return) across states
        means = self.model.means_[:, 0]  # shape (n_states,)
        # argsort descending: highest mean = state 0 (Strong Bull)
        self.regime_map = list(np.argsort(means)[::-1])

    def _original_state(self, regime_idx: int) -> int:
        """Convert semantic regime index → original model state index."""
        return self.regime_map[regime_idx]

    def _semantic_state(self, original_idx: int) -> int:
        """Convert original model state index → semantic regime index."""
        return self.regime_map.index(original_idx)

    # ── Decoding ──────────────────────────────────────────────────────────────

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding: returns array of semantic regime indices (0–6).
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet.")
        _, states = self.model.decode(X, algorithm="viterbi")
        return np.array([self._semantic_state(s) for s in states])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Forward-backward algorithm: returns posterior state probabilities.
        Shape: (n_bars, n_states), columns ordered by semantic regime index.
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet.")
        # posteriors in original model ordering
        posteriors = self.model.predict_proba(X)  # (n_bars, n_states)
        # Reorder columns to semantic ordering
        reordered = posteriors[:, self.regime_map]
        return reordered

    def get_current_regime(self, X: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Returns (current_regime_idx, regime_probabilities) for the latest bar.
        """
        proba = self.predict_proba(X)
        latest_proba = proba[-1]
        current = int(np.argmax(latest_proba))
        return current, latest_proba

    # ── Transition Matrix ─────────────────────────────────────────────────────

    def get_transition_matrix(self) -> np.ndarray:
        """
        Returns transition matrix A reordered to semantic state ordering.
        Shape: (n_states, n_states), entry [i,j] = P(next regime j | current regime i).
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet.")
        A_orig = self.model.transmat_  # (n_states, n_states)
        n = self.n_states
        A_sem = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                orig_i = self.regime_map[i]
                orig_j = self.regime_map[j]
                A_sem[i, j] = A_orig[orig_i, orig_j]
        return A_sem

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood score of data under model."""
        return float(self.model.score(X))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"HMM saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CryptoHMM":
        obj = joblib.load(path)
        logger.info(f"HMM loaded from {path}")
        return obj

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def log_likelihood(self) -> float:
        return self._log_likelihood

    @property
    def state_means(self) -> np.ndarray:
        """Returns state means in semantic order (first feature = log_return)."""
        if not self._trained:
            return np.zeros((self.n_states, 1))
        return self.model.means_[self.regime_map]
