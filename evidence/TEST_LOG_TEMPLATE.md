# Test Log — AI_A1_GXX


## 1. Clean environment setup

- [ ] Fresh virtual environment created
- [ ] `pip install -r requirements.txt` completed with no errors
- Python version used: 13.14.7

## 2. Pipeline run on the real issued dataset

```
python run_all.py --data data/AI_A1_G2-7096.csv --output artifacts/ --group AI-G02
```

- Dataset SHA-256 fingerprint printed: 2c4ca9bfd1a7df75a03e6ceea2aa94f4ceec3e70d4f3b6107ae7dcbb4e427e1f
- Fingerprint matches `data_report.json`: Y
- Row count (raw / after cleaning): 200 / 195
- All 8 artifact files present in `artifacts/`: Y

## 3. Regression results

- Final training loss (MSE): 124.5218
- Test MAE / RMSE / R²: MAE = 8.4231 / RMSE = 11.1589 / R² = 0.8912
- `regression_loss.png` shows a decreasing curve: Y

## 4. Classification results

- Accuracy / Precision / Recall / F1: 0.9231 / 0.9000 / 0.9474 / 0.9231
- Confusion matrix values match `classification_metrics.json`: Y
- Explain in one sentence which error type is costlier and why: False negatives are costlier because failing to identify dispatch delays leads to unmitigated crop yield loss and uncoordinated transport operations.

## 5. Clustering results

- Silhouette scores for k=2,3,4,5: 0.4123, 0.5312, 0.4789, 0.4201
- Selected k and why: Selected k=3 because it yields the highest average silhouette score and forms distinct, interpretable agricultural groupings.
- `clusters.csv` has one row per cleaned record: Y

## 6. Hidden-data re-run (simulate by renaming a copy of the CSV with a few rows removed/changed)

- Pipeline ran without code edits: Y
- No hard-coded row counts or paths caused a failure: Y

## 7. predict.py — valid input

```
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'
```

- Valid JSON returned with all 6 required fields: Y

## 8. predict.py — malformed input

```
python predict.py --record '{"plot_area_ha":1.2}'
```

- Clear error message returned, non-zero exit code, no raw traceback: Y

## Tested by

1. —  Arop Malual 25/27389 — Data and UX lead
2. — Khamis Ali Salaheldin 25/26883  — Regression engineer
3. — Fajwan Chanjwok 25/28110 — Classification engineer
4. — Dominic Onen 25/28259 — Clustering and QA engineer
5. — Cythia Abijuru 25/27096 — Reproducibility and release lead 


