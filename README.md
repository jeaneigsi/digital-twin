# Production Regime Transition Digital Twin

Prototype inspired by **production planning under uncertainty** — simulating a
production line with nominal and degraded regimes, detecting regime changes
from operational indicators, and adapting the schedule via constraint
programming.

The goal is to connect four ideas: **state representation, regime transition
detection, scheduling under constraints, and decision traceability.**

---

## Dashboard

![Header and KPIs](asset/ScreenShot_20260518013013.jpg)

![Operational Indicators and Regime Detection](asset/ScreenShot_20260518013036.jpg)

![Scheduling Adaptation – Gantt charts](asset/ScreenShot_20260518013051.jpg)

![Raw simulation events](asset/ScreenShot_20260518013108.jpg)

---

## Stack

| Block                    | Tool                  |
| ------------------------ | --------------------- |
| Simulation               | SimPy                 |
| Regime detection         | ruptures + scikit-learn |
| Scheduling               | OR-Tools CP-SAT       |
| Experiment tracking      | MLflow                |
| Dashboard                | Streamlit + Altair    |

## Quick start

```bash
uv sync
uv run python src/simulation.py          # standalone demo
uv run streamlit run src/app.py          # interactive dashboard
```

## Structure

```
src/
  config.py             # Dataclass configuration
  simulation.py         # SimPy production line (3 machines, nominal/degraded)
  features.py           # Window-based operational indicators
  regime_detection.py   # ruptures.Pelt + RandomForestClassifier
  scheduler.py          # OR-Tools CP-SAT job-shop with regime adaptation
  experiments.py        # MLflow experiment tracking
  app.py                # Streamlit dashboard
```
