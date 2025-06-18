# trainML.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib

# CONFIG
PRICE_FILE   = "prices.txt"   # rows = days, cols = instruments
MODEL_PATH   = "model.pkl"
FWD_HORIZON  = 5              # predict 5-day forward return
LOOKBACK_1   = 1
LOOKBACK_5   = 5
LOOKBACK_20  = 20
N_ESTIMATORS = 100
RANDOM_STATE = 42

# 1) load price matrix
prices = np.loadtxt(PRICE_FILE)    # shape (days, inst)

# 2) build feature tensors
rets1   = np.vstack([np.zeros(prices.shape[1]), np.diff(prices, axis=0)]) / prices
rets5   = (prices[LOOKBACK_5:] / prices[:-LOOKBACK_5] - 1.0)
rets20  = (prices[LOOKBACK_20:] / prices[:-LOOKBACK_20] - 1.0)
# pad to full length
rets5   = np.vstack([np.zeros((LOOKBACK_5, prices.shape[1])), rets5])
rets20  = np.vstack([np.zeros((LOOKBACK_20, prices.shape[1])), rets20])

# 20-day rolling vol of 1d rets
vol20 = np.zeros_like(prices)
for t in range(LOOKBACK_20, prices.shape[0]):
    vol20[t] = np.std(rets1[t-LOOKBACK_20+1:t+1], axis=0)

# cross-sectional rank of 1-day rets each day
cs_rank = np.zeros_like(prices)
for t in range(prices.shape[0]):
    cs_rank[t] = np.argsort(np.argsort(rets1[t])) / float(prices.shape[1]-1)

# label: forward 5-day return
fwd5 = np.vstack([prices[FWD_HORIZON:] / prices[:-FWD_HORIZON] - 1.0,
                  np.zeros((FWD_HORIZON, prices.shape[1]))])

# 3) stack into (samples × features) and label vector
# each sample = one (day,inst)
days, insts = prices.shape
# features: [r1, r5, r20, vol20, cs_rank]
X = np.stack([rets1, rets5, rets20, vol20, cs_rank], axis=2)        # (days,insts,5)
X = X.reshape(days*insts, 5)                                       # (days*insts, 5)
y = fwd5.reshape(days*insts)                                       # (days*insts,)

# drop any rows where label nan/inf (should be none)
mask = np.isfinite(y)
X, y = X[mask], y[mask]

# 4) time-series CV and training
tscv = TimeSeriesSplit(n_splits=5)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf",      RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                      n_jobs=-1,
                                      random_state=RANDOM_STATE))
])

# cross‐validate
mse_scores = -cross_val_score(pipe, X, y, cv=tscv, scoring="neg_mean_squared_error")
print(f"CV MSE: {mse_scores.mean():.5f} ± {mse_scores.std():.5f}")

# fit on full data
pipe.fit(X, y)
joblib.dump(pipe, MODEL_PATH)
print(f"Trained pipeline saved to {MODEL_PATH}")
