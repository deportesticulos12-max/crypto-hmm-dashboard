"""
Scenario simulation engine.
Perturbs the current feature vector and re-runs HMM forward pass
to show how regime probabilities shift under hypothetical market events.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from config.settings import SCENARIOS, REGIME_NAMES


@dataclass
class ScenarioResult:
    scenario_name: str
    description: str
    icon: str
    baseline_proba: np.ndarray        # (7,) before scenario
    shocked_proba: np.ndarray         # (7,) after scenario
    proba_delta: np.ndarray           # shocked - baseline
    dominant_regime_baseline: str
    dominant_regime_shocked: str
    regime_changed: bool
    # Directional shifts
    p_up_baseline: float
    p_up_shocked: float
    p_down_baseline: float
    p_down_shocked: float


class ScenarioSimulator:
    """
    Simulates hypothetical market scenarios by shocking input features
    and re-running the HMM forward algorithm.
    """

    def __init__(self, hmm_model, feature_engineer, n_states: int = 7):
        self.hmm = hmm_model
        self.fe = feature_engineer
        self.n_states = n_states

    def _shock_features(
        self,
        X_current: np.ndarray,
        shock_spec: dict,
        feature_names: list,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Apply additive Gaussian shocks to specified features.
        X_current: (n_bars, n_feats) or (n_feats,) for single bar.
        shock_spec: {feature_name_substring: shock_sigma_multiplier}
        intensity: global scale factor (0.5 = half-strength, 2.0 = double)
        """
        X = X_current.copy()
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        for feat_substr, sigma_mult in shock_spec.items():
            # Find matching feature columns (substring match)
            matching = [
                i for i, name in enumerate(feature_names)
                if feat_substr in name
            ]
            for idx in matching:
                # The features are already standardized, so 1.0 = 1 std dev
                X[-1, idx] += sigma_mult * intensity

        return X[0] if single else X

    def run_scenario(
        self,
        X_full: np.ndarray,
        feature_names: list,
        scenario_name: str,
        intensity: float = 1.0,
        directional_engine=None,
        transition_matrix: Optional[np.ndarray] = None,
    ) -> ScenarioResult:
        """
        Run a named scenario from settings.SCENARIOS.
        Returns a ScenarioResult with before/after regime probabilities.
        """
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario_cfg = SCENARIOS[scenario_name]
        shock_spec = scenario_cfg["shocks"]

        # ── Baseline: current regime probabilities ────────────────────────────
        baseline_proba_all = self.hmm.predict_proba(X_full)
        baseline_proba = baseline_proba_all[-1]

        # ── Shocked: perturb last bar and re-run forward pass ─────────────────
        X_shocked = self._shock_features(X_full, shock_spec, feature_names, intensity)
        shocked_proba_all = self.hmm.predict_proba(X_shocked)
        shocked_proba = shocked_proba_all[-1]

        # ── Directional shifts ────────────────────────────────────────────────
        if directional_engine is not None and transition_matrix is not None:
            recent = X_full[-5:, 0] if X_full.shape[0] >= 5 else X_full[:, 0]
            base_fc = directional_engine.forecast(baseline_proba, transition_matrix, recent)
            shock_fc = directional_engine.forecast(shocked_proba, transition_matrix, recent)
            p_up_base, p_up_shock = base_fc.p_up, shock_fc.p_up
            p_down_base, p_down_shock = base_fc.p_down, shock_fc.p_down
        else:
            p_up_base = p_up_shock = p_down_base = p_down_shock = 0.0

        dominant_base = REGIME_NAMES[int(np.argmax(baseline_proba))]
        dominant_shock = REGIME_NAMES[int(np.argmax(shocked_proba))]

        return ScenarioResult(
            scenario_name=scenario_name,
            description=scenario_cfg["description"],
            icon=scenario_cfg["icon"],
            baseline_proba=baseline_proba,
            shocked_proba=shocked_proba,
            proba_delta=shocked_proba - baseline_proba,
            dominant_regime_baseline=dominant_base,
            dominant_regime_shocked=dominant_shock,
            regime_changed=(dominant_base != dominant_shock),
            p_up_baseline=p_up_base,
            p_up_shocked=p_up_shock,
            p_down_baseline=p_down_base,
            p_down_shocked=p_down_shock,
        )

    def run_custom_scenario(
        self,
        X_full: np.ndarray,
        feature_names: list,
        custom_shocks: dict,
        intensity: float = 1.0,
        directional_engine=None,
        transition_matrix: Optional[np.ndarray] = None,
    ) -> ScenarioResult:
        """Run a custom scenario with user-defined feature shocks."""
        baseline_proba_all = self.hmm.predict_proba(X_full)
        baseline_proba = baseline_proba_all[-1]

        X_shocked = self._shock_features(X_full, custom_shocks, feature_names, intensity)
        shocked_proba_all = self.hmm.predict_proba(X_shocked)
        shocked_proba = shocked_proba_all[-1]

        if directional_engine is not None and transition_matrix is not None:
            recent = X_full[-5:, 0] if X_full.shape[0] >= 5 else X_full[:, 0]
            base_fc = directional_engine.forecast(baseline_proba, transition_matrix, recent)
            shock_fc = directional_engine.forecast(shocked_proba, transition_matrix, recent)
            p_up_base, p_up_shock = base_fc.p_up, shock_fc.p_up
            p_down_base, p_down_shock = base_fc.p_down, shock_fc.p_down
        else:
            p_up_base = p_up_shock = p_down_base = p_down_shock = 0.0

        return ScenarioResult(
            scenario_name="Custom",
            description="User-defined scenario",
            icon="🎛️",
            baseline_proba=baseline_proba,
            shocked_proba=shocked_proba,
            proba_delta=shocked_proba - baseline_proba,
            dominant_regime_baseline=REGIME_NAMES[int(np.argmax(baseline_proba))],
            dominant_regime_shocked=REGIME_NAMES[int(np.argmax(shocked_proba))],
            regime_changed=(
                REGIME_NAMES[int(np.argmax(baseline_proba))] !=
                REGIME_NAMES[int(np.argmax(shocked_proba))]
            ),
            p_up_baseline=p_up_base,
            p_up_shocked=p_up_shock,
            p_down_baseline=p_down_base,
            p_down_shocked=p_down_shock,
        )
