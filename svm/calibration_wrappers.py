"""Calibration wrappers required to load the frozen SVM artifact."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


class PredictProbaSigmoidCalibrator:
    """Apply an OOF-fitted sigmoid to a frozen sklearn probability model."""

    def __init__(self, base_model, intercept: float, slope: float):
        self.base_model = base_model
        self.intercept = float(intercept)
        self.slope = float(slope)
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, x):
        raw = np.asarray(self.base_model.predict_proba(x)[:, 1], dtype=float)
        positive = expit(self.intercept + self.slope * raw)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

