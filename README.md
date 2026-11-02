# MLOps Lab 3 MLflow Model Registry: Aliases & a Conditional Promotion Gate

**Course:** DADS 7305 MLOps, Northeastern University

**Original lab:** [`raminmohammadi/MLOps/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab2`](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab2) (Wine Quality Prediction)

## Overview

A binary wine-quality classifier (red + white wine datasets merged,
"high quality" = quality ≥ 7), trained two ways a baseline RandomForest
and a hyperparameter-tuned XGBoost challenger tracked and registered
with MLflow, and served via a REST endpoint. The point of this lab is
the **MLflow Model Registry promotion mechanism itself**, not the wine
data or the model types.

## What's different from the original lab

### 1. Migrated off the deprecated stages API

The original lab promotes models like this:

```python
client.transition_model_version_stage(name=model_name, version=version, stage="Production")
```

The `Staging`/`Production`/`Archived` **stages** concept is deprecated in
current MLflow, replaced by **aliases and tags**. This submission uses
the current API throughout:

```python
client.set_registered_model_alias(model_name, "production", version)
```

Confirmed working against MLflow **3.1.4** the Models tab correctly
shows a `production` alias badge on whichever version holds it.

### 2. Added a conditional promotion gate

The original lab **always** promotes whatever model was just registered,
with no check on whether it's actually better than what's currently
serving. This submission's `promote_model.py` fetches the AUC of the
model version currently holding the `production` alias, fetches the
candidate's own AUC (both logged during training), and only moves the
alias if the candidate's AUC is strictly higher. The one exception is
the bootstrap case if nothing holds the `production` alias yet (e.g.
the very first model ever registered), there's nothing to compare
against, so it's promoted unconditionally.

This mirrors the same "don't ship a worse model" pattern as this
course's Docker lab validation gate, applied here to the registry layer
instead of a container build.

**Tested evidence, this exact session:**

| Step                      | Candidate                | AUC    | Current production | AUC    | Result                         |
| ------------------------- | ------------------------ | ------ | ------------------ | ------ | ------------------------------ |
| Baseline, first ever      | v1 (RandomForest)        | 0.8539 | _(none)_           | —      | **PROMOTED** (bootstrap)       |
| Rerun, same version       | v1                       | 0.8539 | v1                 | 0.8539 | **Nothing to do** (idempotent) |
| Challenger beats baseline | v2 (XGBoost, tuned)      | 0.8975 | v1                 | 0.8539 | **PROMOTED**                   |
| Deliberately weak model   | v3 (1-tree RandomForest) | 0.5713 | v2                 | 0.8975 | **REFUSED**, alias unchanged   |

```
Candidate: version 3, AUC = 0.5713
Current production: version 2, AUC = 0.8975
REFUSED: version 3 (AUC 0.5713) does NOT beat current production version 2 (AUC 0.8975). Production alias left unchanged.
```

### Scope decision: Spark batch inference dropped

The original notebook includes a batch-inference section using PySpark
(`SparkTrials` for the hyperparameter search, and a Spark UDF for batch
scoring). This requires a full Spark/Java runtime unrelated to the
registry mechanism this submission focuses on, so it was left out of
scope. The hyperparameter search instead uses hyperopt's plain
`Trials()`, running sequentially — slower for a large search, but
requires nothing beyond `pip install -r requirements.txt`. `max_evals`
was reduced from the original's 96 to 20 accordingly. Real-time serving
via `mlflow models serve` + a REST request was kept, since it's
lightweight and directly tests that the promoted model is actually
servable.

### Two bugs found and fixed during testing

- **Tracking URI mismatch.** Running a training script as its own
  process (writing to a local file store) alongside a separately-running
  `mlflow ui` server caused an `mlflow-artifacts URI... must be a valid
http or https URI` error, because the two processes disagreed on how
  artifacts should be stored. Fixed by explicitly calling
  `mlflow.set_tracking_uri("http://localhost:5001")` in every script, so
  all of them talk to the same server consistently.
- **Alias-detection bug.** An early version of `promote_model.py` used a
  bare `except Exception` when checking for an existing `production`
  alias, which silently swallowed an unrelated error and caused the
  script to always think no alias existed, even after one was
  successfully set. Fixed by catching `MlflowException` specifically and
  only treating a genuine "not found" as the no-alias case; anything
  else is now surfaced loudly instead of hidden.

**Known limitation (not a bug in this code):** immediately after calling
`set_registered_model_alias`, the very next read via
`get_model_version_by_alias` occasionally still returned the previous
alias holder once, before correctly reflecting the update on the next
call. This is a read-after-write lag in MLflow's lightweight local `ui`
server (documented as a single-user, non-production tool), observed
during testing and not something this submission's logic caused.

## Prerequisites

- **Python 3.9+** (tested on 3.9.6)
- **macOS only:** XGBoost requires the OpenMP runtime, which macOS does
  not ship by default:
  ```bash
  brew install libomp
  ```
  (If you don't have Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- No other local installs needed — everything else installs via pip.

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested dependency versions (installed via `requirements.txt`, no pins
needed beyond MLflow): mlflow 3.1.4, scikit-learn 1.6.1, xgboost 2.1.4,
hyperopt 0.2.7, pandas 2.3.3.

Data files (`winequality-red.csv`, `winequality-white.csv`) must be in
a `data/` folder at the project root.

## Running it

**1. Start the MLflow tracking server** (separate terminal tab, leave running):

```bash
mlflow ui --port=5001
```

Open `http://localhost:5001` to confirm it loads.

**2. Train and promote the baseline** (new terminal tab, venv active, from project root):

```bash
python src/train_baseline.py
python src/promote_model.py
```

Expected: trains in a few seconds, prints an AUC around 0.85, registers
as version 1, and promotes it (bootstrap case — nothing exists to
compare against yet).

**3. Train and promote the challenger:**

```bash
python src/train_challenger.py
python src/promote_model.py
```

Expected: runs 20 sequential hyperopt trials (~1 minute), prints the
best trial's AUC (typically ~0.89–0.90), registers as version 2. The
promotion step should report `PROMOTED` since this consistently beats
the baseline.

**4. Demonstrate the gate refusing a worse model:**

```bash
python src/train_weak_model.py
python src/promote_model.py --version <the version number it just printed>
```

Expected: `REFUSED`, with both AUC values printed, and the `production`
alias left on the challenger version.

**5. Serve the promoted model and test it:**

```bash
mlflow models serve --env-manager=local -m "models:/wine_quality@production" -h 0.0.0.0 -p 5002
```

In another terminal (venv active):

```bash
python src/test_inference.py
```

Expected: prints 5 true labels and 5 corresponding predicted
probabilities from the live served model.

## What a fully successful run looks like, end to end

1. `mlflow ui` running, reachable in a browser
2. Baseline trained → registered → promoted (bootstrap)
3. Challenger trained → registered → promoted (beats baseline)
4. Weak model trained → registered → **refused** by the gate
5. MLflow UI's Models tab shows the `production` alias on the
   challenger's version, not the weak model's
6. `mlflow models serve` running, and `test_inference.py` returns real
   prediction probabilities with no errors

## Project structure

```
.
├── data/
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── src/
│   ├── data_prep.py         # shared data loading/splitting
│   ├── train_baseline.py    # trains + registers the RandomForest baseline
│   ├── train_challenger.py  # hyperopt-tuned XGBoost (no Spark)
│   ├── train_weak_model.py  # deliberately bad model, for testing the gate's refusal
│   ├── promote_model.py     # CORE MODIFICATION: aliases + conditional gate
│   └── test_inference.py    # hits the served model's REST endpoint
└── requirements.txt
```

## Author

Sree Rachnae Shyam ([@shyamrachna25](https://github.com/shyamrachna25))
