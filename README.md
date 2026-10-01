# AI_A1_G02 — Musanze Cooperative Harvest and Dispatch Decision Lab

**G02-7096:** 
**Group members:** 
1. —  Arop Malual 25/27389 — Data and UX lead
2. — Khamis Ali Salaheldin 25/26883  — Regression engineer
3. — Fajwan Chanjwok 25/28110 — Classification engineer
4. — Dominic Onen 25/28259 — Clustering and QA engineer
5. — Cythia Abijuru 25/27096 — Reproducibility and release lead 

**Repository URL:** *(paste your GitHub URL here)*
**Final commit hash:** *(paste after your last commit before submission)*
**Dataset SHA-256 fingerprint:** bc33697b34016fe1ff90555c1899b448cb4729fa463477bedc2c74a69dab5d66
---

## ⚠️ Before you do anything else

`data/AI_A1_G2-7096.csv` in this folder is a **synthetic placeholder** generated
only so the pipeline is testable end-to-end. **You must replace it with your
real lecturer-issued dataset**, keeping the exact filename `AI_A1_GXX.csv`
(with your real group number). Do not submit the placeholder file — the
assignment brief explicitly prohibits generated data in the submission, and
the assessor will check the fingerprint against the file they issued you.

## Tested environment

- Python 3.14.7 
- OS: window 

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       
pip install -r requirements.txt
```

## Running the pipeline

```bash
python run_all.py --data data/AI_A1_G2-7096.csv --output artifacts/ --group AI-GXX
```

This prints progress for each of the four stages (data pipeline, regression,
classification, clustering), prints the dataset's SHA-256 fingerprint twice
(once per stage, once in the summary), and writes everything listed below
into `artifacts/` and `models/`.

## Running a prediction

```bash
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}' --group AI-G2-7096
```

`predict.py` requires `run_all.py` to have been run first (it loads the
saved model files from `models/`). It prints one JSON object with the
regression prediction, classification prediction + probability, cluster
label, group code, and model version.

**Malformed input example** (missing fields — this is expected to fail
with a clear error on stderr and exit code 1, not a traceback):

```bash
python predict.py --record '{"plot_area_ha":1.2}'
```

## Expected outputs

After `run_all.py` completes, `artifacts/` contains:

| File | Contents |
|---|---|
| `data_report.json` | row/feature counts, missing values, descriptive stats, group02, SHA-256 fingerprint |
| `regression_metrics.json` | seed, learning rate, iterations, MAE, RMSE, R², sample predictions |
| `regression_loss.png` | training loss curve over gradient descent iterations |
| `classification_metrics.json` | accuracy, precision, recall, F1, confusion matrix, cost-of-errors explanation |
| `confusion_matrix.png` | confusion matrix heatmap |
| `clustering_metrics.json` | silhouette scores for k=2..5, selected k, cluster sizes, caution note |
| `clusters.csv` | `record_id`, `cluster_label` for every row |
| `cluster_plot.png` | PCA-projected scatter plot colored by cluster |

`models/` contains the three saved model bundles (`regression_model.joblib`,
`classification_model.joblib`, `clustering_model.joblib`) that `predict.py`
loads.

## Design notes / why these choices

- **Regression**: implemented entirely with NumPy (no sklearn estimator) —
  batch gradient descent on standardized features, as required. Feature
  scaling uses `sklearn.preprocessing.StandardScaler` since the restriction
  in the brief is specifically on the regression *estimator*, not on the
  scaler, and the scaler is fit on the training split only.


- **Classification**: logistic regression, chosen for interpretability —
  its coefficients can be explained directly to non-technical cooperative
  staff ("more distance increases dispatch-attention risk," etc.), which
  matters more here than squeezing out marginal accuracy from a black-box
  model.

- **Clustering**: k is chosen by silhouette score across k=2..5 rather than
  fixed, so the "best" k adapts to whatever the actual issued dataset looks
  like. Targets (`actual_yield_kg`, `dispatch_attention`) and the identifier
  are explicitly excluded from the clustering input — only the six feature
  columns are used.

- **Reproducibility**: a single `RANDOM_SEED` constant in `src/utils.py` is
  used for the train/test splits, gradient descent initialization, and
  KMeans initialization, and is recorded in every metrics JSON file.

## Known limitations

- The classifier's precision/recall tradeoff hasn't been tuned past a
  default 0.5 threshold — see `classification_metrics.json` for the
  as-reported numbers on your actual dataset; these will change once you
  swap in the real CSV.
- Clustering silhouette scores tend to be modest (0.1–0.3 is common) for
  cooperative operating-profile data like this — a low score doesn't mean
  the code is broken, it means the natural groupings in the data are
  genuinely soft. The `caution` field in `clustering_metrics.json` reflects
  this.
- `predict.py` validates that the six required fields are present and
  numeric; it does not currently range-check values (e.g. a negative
  `distance_km`). Extending `validate_record()` in `src/utils.py` to add
  range checks would be a reasonable improvement if your group wants to
  go further.

## AI usage disclosure

See `evidence/AI_USE.md`. Per the assignment brief, generative AI assistance
is permitted for explanation, brainstorming, and debugging when disclosed —
it does not substitute for each member being able to explain and modify
their own section live.
