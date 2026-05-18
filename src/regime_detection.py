from __future__ import annotations

import logging
from typing import override

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "queue_length_mean",
    "processing_time_mean",
    "defect_rate",
    "availability",
]


class RegimeDetector:
    """Detect regime transitions and classify operational windows.

    1. ``ruptures.Pelt`` locates change-points in the feature time-series.
    2. A ``RandomForestClassifier`` labels each window as *nominal* or
       *degraded*.
    """

    def __init__(
        self,
        model: str = "rbf",
        penalty: float = 3.0,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.penalty = penalty
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_change_points(self, features: pd.DataFrame) -> list[int]:
        """Return the indices (row positions) of detected change-points."""
        signal = features[FEATURE_COLUMNS].values.astype(np.float64)
        if signal.shape[0] < 2:
            return []

        algo = rpt.Pelt(model=self.model).fit(signal)
        result = algo.predict(pen=self.penalty)
        return result[:-1]  # drop the trailing sentinel

    def fit(self, features: pd.DataFrame) -> RegimeDetector:
        """Train the regime classifier on labelled feature data."""
        X = features[FEATURE_COLUMNS].values
        y = (features["regime"] != "nominal").astype(int).values  # 0 = nominal, 1 = degraded

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.classifier.fit(X_scaled, y)

        logger.info("Classifier trained on %d samples", len(X))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime for each window (0 = nominal, 1 = degraded)."""
        X = features[FEATURE_COLUMNS].values
        X_scaled = self.scaler.transform(X.astype(np.float64))
        return self.classifier.predict(X_scaled)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        X = features[FEATURE_COLUMNS].values
        X_scaled = self.scaler.transform(X.astype(np.float64))
        return self.classifier.predict_proba(X_scaled)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def evaluate(self, features: pd.DataFrame) -> dict:
        """Train on 70 % of the data and report held-out metrics."""
        train, test = train_test_split(features, test_size=0.3, random_state=self.random_state, stratify=features["regime"])
        self.fit(train)
        y_true = (test["regime"] != "nominal").astype(int).values
        y_pred = self.predict(test)

        report = classification_report(y_true, y_pred, target_names=["nominal", "degraded"], output_dict=True)
        logger.info("Accuracy: %.3f", report["accuracy"])
        return report


def run_demo(events: pd.DataFrame) -> dict:
    from src.features import extract_from_events

    features = extract_from_events(events, window_size=50.0)
    detector = RegimeDetector()
    cp = detector.detect_change_points(features)
    logger.info("Detected %d change-points at indices %s", len(cp), cp)
    return detector.evaluate(features)
