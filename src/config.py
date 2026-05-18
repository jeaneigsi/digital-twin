from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MachineConfig:
    name: str
    cycle_time_mean: float
    cycle_time_std: float
    defect_rate: float
    mttf: float
    mttr: float


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    machines: list[MachineConfig]


@dataclass(frozen=True)
class SimulationConfig:
    regimes: list[RegimeConfig]
    inter_arrival_time_mean: float
    random_seed: int = 42
    window_size: float = 50.0

    @property
    def machines_count(self) -> int:
        return len(self.regimes[0].machines)

    def machine_config(self, regime_idx: int, machine_idx: int) -> MachineConfig:
        return self.regimes[regime_idx].machines[machine_idx]


def default_config() -> SimulationConfig:
    return SimulationConfig(
        regimes=[
            RegimeConfig(
                name="nominal",
                machines=[
                    MachineConfig("M1", cycle_time_mean=5.0, cycle_time_std=1.0, defect_rate=0.02, mttf=200.0, mttr=10.0),
                    MachineConfig("M2", cycle_time_mean=4.0, cycle_time_std=0.8, defect_rate=0.01, mttf=250.0, mttr=8.0),
                    MachineConfig("M3", cycle_time_mean=6.0, cycle_time_std=1.2, defect_rate=0.03, mttf=180.0, mttr=12.0),
                ],
            ),
            RegimeConfig(
                name="degraded",
                machines=[
                    MachineConfig("M1", cycle_time_mean=8.0, cycle_time_std=2.5, defect_rate=0.08, mttf=80.0, mttr=20.0),
                    MachineConfig("M2", cycle_time_mean=6.5, cycle_time_std=2.0, defect_rate=0.06, mttf=100.0, mttr=15.0),
                    MachineConfig("M3", cycle_time_mean=9.0, cycle_time_std=3.0, defect_rate=0.10, mttf=70.0, mttr=25.0),
                ],
            ),
        ],
        inter_arrival_time_mean=6.0,
        random_seed=42,
        window_size=50.0,
    )
