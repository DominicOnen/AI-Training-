# Test Log — AI_A1_GXX


## 1. Clean environment setup

- [ ] Fresh virtual environment created
- [ ] `pip install -r requirements.txt` completed with no errors
- Python version used: _______

## 2. Pipeline run on the real issued dataset

```
python run_all.py --data data/AI_A1_GXX.csv --output artifacts/ --group AI-GXX
```

- Dataset SHA-256 fingerprint printed: _______________________________
- Fingerprint matches `data_report.json`: Y / N
- Row count (raw / after cleaning): _____ / _____
- All 8 artifact files present in `artifacts/`: Y / N

## 3. Regression results

- Final training loss (MSE): _______
- Test MAE / RMSE / R²: _______ / _______ / _______
- `regression_loss.png` shows a decreasing curve: Y / N

## 4. Classification results

- Accuracy / Precision / Recall / F1: ___ / ___ / ___ / ___
- Confusion matrix values match `classification_metrics.json`: Y / N
- Explain in one sentence which error type is costlier and why: _______

## 5. Clustering results

- Silhouette scores for k=2,3,4,5: ___, ___, ___, ___
- Selected k and why: _______
- `clusters.csv` has one row per cleaned record: Y / N

## 6. Hidden-data re-run (simulate by renaming a copy of the CSV with a few rows removed/changed)

- Pipeline ran without code edits: Y / N
- No hard-coded row counts or paths caused a failure: Y / N

## 7. predict.py — valid input

```
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'
```

- Valid JSON returned with all 6 required fields: Y / N

## 8. predict.py — malformed input

```
python predict.py --record '{"plot_area_ha":1.2}'
```

- Clear error message returned, non-zero exit code, no raw traceback: Y / N

## Tested by

| Name | Registration number | Role | Date |


