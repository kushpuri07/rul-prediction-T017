# Remaining Useful Life Prediction for Industrial Engines

Predictive maintenance on the NASA C-MAPSS turbofan dataset (subset FD001).
**Phase 1 (this repo state):** engineered degradation features + classical ML.
**Phase 2 (planned):** LSTM / temporal CNN on raw sensor windows, with an optional hybrid, on the same engine-level split.

## Problem
Given the sensor history of an engine up to cycle *t*, predict its Remaining Useful Life (RUL): the number of operating cycles left before failure. This is a regression task.

## Dataset
NASA C-MAPSS FD001: 100 training engines (run to failure), 100 test engines (cut off early), 21 sensors, 3 operating settings, one operating condition and one fault mode.
Source: NASA Prognostics Data Repository, "Turbofan Engine Degradation Simulation Data Set" (Saxena and Goebel, 2008).
The data files are not committed. See `data/README.md` for where to place them.

## Method
1. **Data checks and EDA:** missing values, duplicates, cycle continuity, lifetime and RUL distributions, sensor trends.
2. **Target:** RUL = last cycle − current cycle for each training engine (uncapped for the main model).
3. **Sensor selection:** drop sensors with training std < 0.01 (s1, s5, s6, s10, s16, s18, s19), leaving 14.
4. **Features (57):** cycle, 14 raw sensors, and per sensor a rolling mean, rolling slope (window 5) and health (drift from the engine's own first 10 cycles). All features are causal.
5. **Evaluation:** 5-fold `GroupKFold` on engine ID over all 100 training engines (mean ± std). Final models are retrained on all training engines and scored once on the 100 test engines at their last cycle.
6. **Models:** naive mean baseline, Elastic Net, Random Forest.
7. **Metrics:** RMSE and MAE, plus MAE for early (RUL ≥ 50) and late (RUL < 50) stages.
8. **Secondary comparison:** RUL capped at 125, scored on uncapped RUL.

## Results
Random Forest is best on every metric. CV = 5-fold GroupKFold mean over the 100 training engines; Test = last cycle of each of the 100 test engines. Early = true RUL at least 50.

| Model | Data | RMSE | MAE | MAE early | MAE late |
|---|---|---|---|---|---|
| Naive (mean RUL) | CV mean | 68.87 | 56.86 | 48.40 | 83.31 |
| Elastic Net | CV mean | 40.04 | 30.63 | 32.96 | 23.35 |
| Random Forest | CV mean | 34.78 | 22.79 | 28.47 | 5.04 |
| Naive (mean RUL) | Test | 52.62 | 40.88 | 21.44 | 86.24 |
| Elastic Net | Test | 27.98 | 23.68 | 25.02 | 20.55 |
| Random Forest | Test | 23.34 | 15.58 | 20.54 | 4.02 |

CV and test scores are not directly comparable: CV scores every row of held-out engines across their whole lifetime, while the test score uses one cycle per engine. Capping RUL at 125 scored worse on uncapped RUL (CV RMSE 46.40 vs 34.78), so the main model is uncapped.

## How to run
```bash
git clone <repo-url>
cd rul-prediction-cmapss
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in data/ (see data/README.md)
jupyter notebook notebooks/01_phase1_ml_baseline.ipynb
```
In Google Colab, upload the three files to the session and run all cells. The notebook reads the files from its working directory; edit the three `read_csv` paths to `../data/...` if you run it from `notebooks/`.


## Limitations
Fixed model settings (no tuning); window size chosen on the same engines used for model comparison; baseline length and the late threshold set by hand; test score is one cycle per engine; FD001 only.

## Team
Kush Puri 
Aadit Vachher 
Vachan Gupta 
Aabir Sarkar
Kushagra Maheshwari


## References
- Saxena, A., and Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. NASA Prognostics Data Repository.
- Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
- Breiman, L. (2001). Random Forests. Machine Learning.
- Zou, H., and Hastie, T. (2005). Regularization and variable selection via the elastic net. JRSS B.
