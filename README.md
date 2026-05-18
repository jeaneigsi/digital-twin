# Production Regime Transition Digital Twin

This project is a small prototype inspired by production planning under uncertainty.

It simulates a production line with nominal and degraded regimes, detects regime
changes from operational indicators, and adapts a simple production schedule
using constraint programming.

The goal is not to solve a full industrial problem, but to connect four ideas:
state representation, regime transition detection, scheduling under constraints,
and decision traceability.

## Stack

| Block                    | Tool                  |
| ------------------------ | --------------------- |
| Simulation               | SimPy                 |
| Regime detection         | ruptures + scikit-learn |
| Scheduling               | OR-Tools CP-SAT       |
| Experiment tracking      | MLflow                |
| Dashboard                | Streamlit             |

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
