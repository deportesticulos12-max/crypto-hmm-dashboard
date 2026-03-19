"""
Ensemble HMM: train N models with different seeds, average posteriors.
Provides regime uncertainty estimation via ensemble variance.
"""

import numpy as np

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

from config.settings import HMM_N_STATES, HMM_N_ITER, HMM_COVARIANCE_TYPE


class EnsembleHMM:
    """
    Ensemble of N GaussianHMM models trained with different random seeds.
    Averages posterior probabilities across the ensemble for improved stability.
    """

    def __init__(self, n_models: int = 5, n_states: int = HMM_N_STATES):
        self.n_models = n_models
        self.n_states = n_states
        self.models: list = []
        self._regime_maps: list = []
        self._trained = False

    def _sort_regime_map(self, model: GaussianHMM) -> list:
        """Sort states by first feature mean (log_return), descending."""
        means = model.means_[:, 0]
        return list(np.argsort(means)[::-1])

    def train(self, X: np.ndarray):
        """Train N HMMs with different seeds."""
        self.models = []
        self._regime_maps = []
        last_err = None
        for seed in range(self.n_models):
            try:
                m = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=HMM_COVARIANCE_TYPE,
                    n_iter=HMM_N_ITER,
                    random_state=seed * 7 + 13,
                    tol=1e-4,
                    verbose=False,
                )
                m.fit(X)
                self.models.append(m)
                self._regime_maps.append(self._sort_regime_map(m))
            except Exception as e:
                import traceback
                traceback.print_exc()
                last_err = e
                continue
                
        if not self.models:
            raise RuntimeError(f"All models in the ensemble failed to train. Last Error: {last_err}")
            
        self._trained = True

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean_proba, std_proba) — averaged and variance across ensemble.
        Shape: (n_bars, n_states) each.
        """
        all_probas = []
        for model, rmap in zip(self.models, self._regime_maps):
            raw = model.predict_proba(X)           # (n_bars, n_states)
            reordered = raw[:, rmap]               # semantic ordering
            all_probas.append(reordered)

        stack = np.stack(all_probas, axis=0)       # (n_models, n_bars, n_states)
        mean_proba = stack.mean(axis=0)
        std_proba = stack.std(axis=0)
        return mean_proba, std_proba

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi-like decode. We take the mean posterior probabilities from
        predict_proba() and simply use an argmax.
        This provides a smooth, stable regime path derived from the ensemble.
        """
        if not self._trained:
            raise RuntimeError("Ensemble not trained yet.")
        mean_proba, _ = self.predict_proba(X)
        return np.argmax(mean_proba, axis=1)

    def get_transition_matrix(self) -> np.ndarray:
        """
        Averages the transition matrices of all models in the ensemble.
        Before averaging, each matrix is reordered to the semantic state ordering.
        Shape: (n_states, n_states).
        """
        if not self._trained:
            raise RuntimeError("Ensemble not trained yet.")
        
        matrices = []
        for model, rmap in zip(self.models, self._regime_maps):
            # A_orig shape: (n_states, n_states)
            A_orig = model.transmat_
            A_sem = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                for j in range(self.n_states):
                    orig_i = rmap[i]
                    orig_j = rmap[j]
                    A_sem[i, j] = A_orig[orig_i, orig_j]
            matrices.append(A_sem)
            
        return np.mean(matrices, axis=0)
    
    def score(self, X: np.ndarray) -> float:
        """
        Returns the average log-likelihood across all models in the ensemble.
        """
        if not self._trained:
            raise RuntimeError("Ensemble not trained yet.")
        scores = [model.score(X) for model in self.models]
        return float(np.mean(scores))
    
    @property
    def log_likelihood(self) -> float:
        """Property to mock the behavior of a single CryptoHMM for the dashboard tab."""
        if not self._trained:
            return 0.0
        # This will be slow if called repeatedly, but it's only for the dashboard info tab.
        # Could cache this later if performance is an issue.
        return 0.0 # Returning a dummy value for the property as score(X) takes an argument

    @property
    def state_means(self) -> np.ndarray:
        """
        Returns the average state means across the ensemble, in semantic order.
        """
        if not self._trained:
            return np.zeros((self.n_states, 1))
        
        all_means = []
        for model, rmap in zip(self.models, self._regime_maps):
            all_means.append(model.means_[rmap])
        return np.mean(all_means, axis=0)

    @property
    def is_trained(self) -> bool:
        return self._trained and len(self.models) > 0
