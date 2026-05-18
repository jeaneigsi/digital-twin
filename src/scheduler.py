from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import override

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    job_id: int
    processing_times: list[int]   # one per machine
    priority: int = 1              # higher = more urgent


@dataclass
class ScheduleResult:
    status: str
    makespan: int
    total_flow_time: float
    total_weighted_completion: float
    schedule: pd.DataFrame          # columns: job_id, machine, start, end


class ProductionScheduler:
    """Job-shop scheduler using OR-Tools CP-SAT.

    Takes a list of jobs (each with per-machine processing times) and
    produces a feasible schedule that minimises the makespan.
    """

    def __init__(self, horizon_factor: float = 2.0) -> None:
        self.horizon_factor = horizon_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, jobs: list[Job], machines_count: int) -> ScheduleResult:
        if not jobs:
            return ScheduleResult("empty", 0, 0.0, 0.0, pd.DataFrame())

        horizon = self._estimate_horizon(jobs, machines_count)
        model = cp_model.CpModel()

        # Variables
        n_jobs = len(jobs)
        n_machines = machines_count
        all_tasks: dict[tuple[int, int], tuple[cp_model.IntVar, cp_model.IntVar, cp_model.IntervalVar]] = {}
        ends: list[cp_model.IntVar] = []

        for j_idx, job in enumerate(jobs):
            for m_idx in range(n_machines):
                start_var = model.new_int_var(0, horizon, f"start_j{j_idx}_m{m_idx}")
                duration = max(1, job.processing_times[m_idx])
                end_var = model.new_int_var(0, horizon, f"end_j{j_idx}_m{m_idx}")
                interval_var = model.new_interval_var(start_var, duration, end_var, f"interval_j{j_idx}_m{m_idx}")
                all_tasks[(j_idx, m_idx)] = (start_var, end_var, interval_var)

            end_var = model.new_int_var(0, horizon, f"end_j{j_idx}")
            ends.append(end_var)

        # Constraints
        for j_idx, job in enumerate(jobs):
            for m_idx in range(n_machines - 1):
                _, end_prev, _ = all_tasks[(j_idx, m_idx)]
                start_next, _, _ = all_tasks[(j_idx, m_idx + 1)]
                model.add(start_next >= end_prev)

        for m_idx in range(n_machines):
            machine_intervals = [all_tasks[(j_idx, m_idx)][2] for j_idx in range(n_jobs)]
            model.add_no_overlap(machine_intervals)

        # Link job-end to last operation
        for j_idx in range(n_jobs):
            _, end_last, _ = all_tasks[(j_idx, n_machines - 1)]
            model.add(ends[j_idx] >= end_last)

        # Objective: minimise makespan
        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, ends)
        model.minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status_code = solver.solve(model)

        status = "optimal" if status_code == cp_model.OPTIMAL else "feasible" if status_code == cp_model.FEASIBLE else "infeasible"
        logger.info("Scheduling %s (%d jobs, %d machines)", status, n_jobs, n_machines)

        if status == "infeasible":
            return ScheduleResult(status, 0, 0.0, 0.0, pd.DataFrame())

        rows = []
        completion_times: dict[int, int] = {}
        for j_idx, job in enumerate(jobs):
            for m_idx in range(n_machines):
                start_var, _, _ = all_tasks[(j_idx, m_idx)]
                start_val = solver.value(start_var)
                end_val = start_val + job.processing_times[m_idx]
                rows.append({
                    "job_id": job.job_id,
                    "machine": f"M{m_idx + 1}",
                    "start": start_val,
                    "end": end_val,
                })
                if m_idx == n_machines - 1:
                    completion_times[job.job_id] = end_val

        total_flow = sum(completion_times.values())
        total_weighted = sum(completion_times[j.job_id] * j.priority for j in jobs)

        return ScheduleResult(
            status=status,
            makespan=int(solver.value(makespan)),
            total_flow_time=total_flow,
            total_weighted_completion=total_weighted,
            schedule=pd.DataFrame(rows).sort_values(["machine", "start"]),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_horizon(self, jobs: list[Job], machines_count: int) -> int:
        total = int(sum(sum(j.processing_times) for j in jobs) * self.horizon_factor)
        return max(total, 1)


# ------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------

def generate_jobs(n: int, machines_count: int, rng: np.random.Generator | None = None) -> list[Job]:
    """Synthetic job generator for scheduling demos."""
    rng = rng or np.random.default_rng(42)
    return [
        Job(
            job_id=i,
            processing_times=[int(rng.integers(2, 10)) for _ in range(machines_count)],
            priority=int(rng.integers(1, 4)),
        )
        for i in range(n)
    ]


def adapt_jobs_for_regime(jobs: list[Job], regime: int, slowdown: float = 1.4) -> list[Job]:
    """Adjust jobs based on estimated regime.

    In degraded mode processing times are scaled up and high-workload
    jobs are prioritised to avoid starving downstream machines.
    """
    if regime == 0:
        return jobs

    return sorted(
        [
            Job(
                job_id=j.job_id,
                processing_times=[max(1, int(t * slowdown)) for t in j.processing_times],
                priority=j.priority,
            )
            for j in jobs
        ],
        key=lambda j: sum(j.processing_times),
        reverse=True,
    )
