from __future__ import annotations

import logging
from pathlib import Path
from typing import override

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import SimulationConfig
from src.features import FeatureExtractor, extract_from_events
from src.regime_detection import RegimeDetector
from src.scheduler import ProductionScheduler, ScheduleResult, adapt_jobs_for_regime, generate_jobs
from src.simulation import ProductionLine, build_regime_schedule

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Wraps the full pipeline in an MLflow experiment for traceability."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._tracking_uri = tracking_uri

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        config: SimulationConfig,
        nominal_duration: float = 500.0,
        degraded_duration: float = 500.0,
        recovery_duration: float = 500.0,
        window_size: float = 50.0,
    ) -> dict:
        mlflow.set_experiment("production-regime-detection")

        with mlflow.start_run():
            mlflow.log_params({
                "nominal_duration": nominal_duration,
                "degraded_duration": degraded_duration,
                "recovery_duration": recovery_duration,
                "window_size": window_size,
                "random_seed": config.random_seed,
            })

            # 1. Simulate
            line = ProductionLine(config)
            schedule = build_regime_schedule(nominal_duration, degraded_duration, recovery_duration)
            total_time = nominal_duration + degraded_duration + recovery_duration
            events = line.run(duration=total_time, regime_schedule=schedule)

            mlflow.log_metric("total_jobs", events["job_id"].nunique())
            mlflow.log_metric("total_events", len(events))

            # 2. Features
            extractor = FeatureExtractor(window_size)
            features = extractor.extract(events)

            # 3. Regime detection
            detector = RegimeDetector(random_state=config.random_seed)
            evaluation = detector.evaluate(features)

            mlflow.log_metric("classifier_accuracy", evaluation["accuracy"])
            mlflow.log_metric("nominal_f1", evaluation["nominal"]["f1-score"])
            mlflow.log_metric("degraded_f1", evaluation["degraded"]["f1-score"])

            # Predict on full dataset
            detector.fit(features)
            predictions = detector.predict(features)
            features["predicted_regime"] = ["nominal" if p == 0 else "degraded" for p in predictions]

            # 4. Scheduling
            n_jobs = min(20, max(3, events["job_id"].nunique() // 5))
            jobs = generate_jobs(n_jobs, config.machines_count)
            scheduler = ProductionScheduler()

            for regime_label in ("nominal", "degraded"):
                regime_int = 0 if regime_label == "nominal" else 1
                adapted = adapt_jobs_for_regime(list(jobs), regime_int)
                result = scheduler.schedule(adapted, config.machines_count)
                mlflow.log_metric(f"makespan_{regime_label}", result.makespan)
                mlflow.log_metric(f"flow_time_{regime_label}", result.total_flow_time)
                mlflow.log_metric(f"schedule_status_{regime_label}", 1 if result.status == "optimal" else 0)

            # Log data as artifacts
            events_csv = Path("events.csv")
            features_csv = Path("features.csv")
            events.to_csv(events_csv, index=False)
            features.to_csv(features_csv, index=False)
            mlflow.log_artifact(str(events_csv))
            mlflow.log_artifact(str(features_csv))
            events_csv.unlink(missing_ok=True)
            features_csv.unlink(missing_ok=True)

            logger.info("MLflow run completed – run_id: %s", mlflow.active_run().info.run_id)
            return evaluation


def run_experiment(config: SimulationConfig | None = None) -> dict:
    from src.config import default_config

    cfg = config or default_config()
    tracker = ExperimentTracker()
    return tracker.run(cfg)
