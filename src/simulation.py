from __future__ import annotations

import logging
import math
import random
from typing import override

import pandas as pd
import simpy

from src.config import MachineConfig, SimulationConfig

logger = logging.getLogger(__name__)


class ProductionLine:
    """Discrete-event simulation of a serial production line using SimPy."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.env = simpy.Environment()
        self.machines = [simpy.Resource(self.env, capacity=1) for _ in range(config.machines_count)]
        self._current_regime_idx: int = 0
        self._events: list[dict] = []
        self._job_counter: int = 0
        self._regime_schedule: list[tuple[float, int]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, duration: float, regime_schedule: list[tuple[float, int]]) -> pd.DataFrame:
        """Run the simulation and return an event-log DataFrame.

        Parameters
        ----------
        duration : float
            Total simulation time.
        regime_schedule : list[tuple[float, int]]
            Sequence of (switch_time, regime_index).  Must start with (0, …).

        Returns
        -------
        pd.DataFrame
            Columns: time, job_id, machine, queue_length, processing_time,
            is_defect, regime, event_type.
        """
        self._regime_schedule = regime_schedule
        self._current_regime_idx = regime_schedule[0][1]

        self.env.process(self._source())
        self.env.process(self._regime_switcher())
        self.env.run(until=duration)

        logger.info("Simulation finished – %d jobs, %d events", self._job_counter, len(self._events))
        return self._to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        return self._to_dataframe()

    @property
    def events(self) -> list[dict]:
        return self._events

    # ------------------------------------------------------------------
    # Processes
    # ------------------------------------------------------------------

    def _source(self) -> simpy.Process:
        mean_iat = self.config.inter_arrival_time_mean
        while True:
            inter_arrival = self.rng.expovariate(1.0 / mean_iat) if mean_iat > 0 else math.inf
            yield self.env.timeout(inter_arrival)
            self._job_counter += 1
            self.env.process(self._job_process(self._job_counter))

    def _job_process(self, job_id: int) -> simpy.Process:
        for m_idx in range(self.config.machines_count):
            resource = self.machines[m_idx]
            cfg = self._active_machine_configs()[m_idx]

            congestion = len(resource.queue) + resource.count

            with resource.request() as req:
                yield req
                yield from self._process_on_machine(job_id, m_idx, cfg, congestion)

    def _process_on_machine(self, job_id: int, m_idx: int, cfg: MachineConfig, congestion: int):
        failure_draw = self.rng.random()
        failure_threshold = cfg.cycle_time_mean / cfg.mttf if cfg.mttf > 0 else 0.0
        failed = failure_draw < failure_threshold

        if failed:
            yield self.env.timeout(self.rng.expovariate(1.0 / cfg.mttr) if cfg.mttr > 0 else 0.0)

        proc_time = max(0.1, self.rng.gauss(cfg.cycle_time_mean, cfg.cycle_time_std))
        yield self.env.timeout(proc_time)

        is_defect = self.rng.random() < cfg.defect_rate

        self._events.append({
            "time": round(self.env.now, 4),
            "job_id": job_id,
            "machine": cfg.name,
            "queue_length": congestion,
            "processing_time": round(proc_time, 4),
            "is_defect": is_defect,
            "machine_available": not failed,
            "regime": self.config.regimes[self._current_regime_idx].name,
        })

    def _regime_switcher(self) -> simpy.Process:
        for switch_time, regime_idx in self._regime_schedule[1:]:
            yield self.env.timeout(switch_time - self.env.now)
            self._current_regime_idx = regime_idx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_machine_configs(self) -> list[MachineConfig]:
        return self.config.regimes[self._current_regime_idx].machines

    def _to_dataframe(self) -> pd.DataFrame:
        if not self._events:
            return pd.DataFrame(columns=[
                "time", "job_id", "machine", "queue_length",
                "processing_time", "is_defect", "machine_available", "regime",
            ])
        df = pd.DataFrame(self._events)
        df["is_defect"] = df["is_defect"].astype(bool)
        df["machine_available"] = df["machine_available"].astype(bool)
        return df.sort_values("time").reset_index(drop=True)


# ------------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------------

def build_regime_schedule(
    nominal_duration: float,
    degraded_duration: float,
    nominal_duration_2: float,
) -> list[tuple[float, int]]:
    """Build a NOMINAL → DEGRADED → NOMINAL schedule."""
    t0 = 0.0
    t1 = t0 + nominal_duration
    t2 = t1 + degraded_duration
    # t3 = t2 + nominal_duration_2  (end of simulation)
    return [(t0, 0), (t1, 1), (t2, 0)]


def run_demo() -> pd.DataFrame:
    """Quick demonstration of the simulation."""
    from src.config import default_config

    cfg = default_config()
    line = ProductionLine(cfg)
    schedule = build_regime_schedule(500.0, 500.0, 500.0)
    df = line.run(duration=1500.0, regime_schedule=schedule)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = run_demo()
    print(df.head(10))
    print(f"\nTotal events: {len(df)}")
    print(df["regime"].value_counts())
