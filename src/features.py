from __future__ import annotations

from typing import override

import numpy as np
import pandas as pd


class FeatureExtractor:
    """Extract aggregated operational indicators from raw simulation events.

    Uses rolling time-windows (not fixed-count windows) so that irregularly
    spaced SimPy events are handled correctly.
    """

    def __init__(self, window_size: float = 50.0) -> None:
        self.window_size = window_size

    def extract(self, events: pd.DataFrame) -> pd.DataFrame:
        """Compute per-window features from the raw event log.

        Parameters
        ----------
        events : pd.DataFrame
            Output of ``ProductionLine.run()``.  Must contain columns:
            time, machine, queue_length, processing_time, is_defect,
            machine_available, regime.

        Returns
        -------
        pd.DataFrame
            One row per time-window with columns:
            time, regime (mode), queue_length_mean, queue_length_std,
            processing_time_mean, processing_time_std, defect_rate,
            availability, throughput.
        """
        if events.empty:
            return pd.DataFrame()

        df = events.copy()
        df["time_bin"] = (df["time"] // self.window_size).astype(int)

        features = df.groupby("time_bin").agg(
            time=("time", "mean"),
            queue_length_mean=("queue_length", "mean"),
            queue_length_std=("queue_length", "std"),
            processing_time_mean=("processing_time", "mean"),
            processing_time_std=("processing_time", "std"),
            defect_rate=("is_defect", "mean"),
            availability=("machine_available", "mean"),
            event_count=("time", "count"),
        ).reset_index(drop=True)

        regime_mode = df.groupby("time_bin")["regime"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        features["regime"] = regime_mode.values

        features["throughput"] = features["event_count"] / self.window_size
        features = features.fillna(0.0)

        columns = [
            "time", "regime", "queue_length_mean", "queue_length_std",
            "processing_time_mean", "processing_time_std", "defect_rate",
            "availability", "throughput",
        ]
        return features[columns]


def extract_from_events(events: pd.DataFrame, window_size: float = 50.0) -> pd.DataFrame:
    return FeatureExtractor(window_size).extract(events)
