# main.py
import numpy as np
import joblib

# CONFIG
MODEL_PATH = "model.pkl"
POS_LIMIT  = 10_000    # $10k cap per instrument
THRESH_STD = 0.5       # min‐signal threshold (in std units)
VOL_LOOK   = 20        # days for vol estimate in trading
TARGET_VOL = 0.01      # target daily vol (~1%)

# load trained pipeline
pipe = joblib.load(MODEL_PATH)

def getMyPosition(prcHist):
    """
    prcHist: array (nInst, t) of historical prices up to today
    returns: integer positions (length nInst)
    """
    # transpose to (t,inst)
    prices = prcHist.T
    days, insts = prices.shape

    # build today’s features for each instrument
    # 1d, 5d, 20d returns
    r1   = np.zeros(insts)
    r5   = np.zeros(insts)
    r20  = np.zeros(insts)
    vol20= np.zeros(insts)
    csr  = np.zeros(insts)

    if days > 1:
        r1 = (prices[-1] / prices[-2] - 1.0)
    if days > 5:
        r5 = (prices[-1] / prices[-6] - 1.0)
    if days > 20:
        r20 = (prices[-1] / prices[-21] - 1.0)
        # vol20
        rets1 = (prices[1:] / prices[:-1] - 1.0)
        vol20 = np.std(rets1[-20:], axis=0)
    # cross‐sectional rank of r1
    csr[np.argsort(np.argsort(r1))] = np.arange(insts) / (insts - 1)

    # assemble feature matrix (insts × 5)
    X = np.stack([r1, r5, r20, vol20, csr], axis=1)

    # predict forward 5d returns
    preds = pipe.predict(X)
    mu, sd = preds.mean(), preds.std()

    # threshold small signals
    sig = np.where(np.abs(preds - mu) > THRESH_STD*sd,
                   np.sign(preds - mu),
                   0.0)

    # raw dollar exposures ∝ signal
    dol = sig * POS_LIMIT
    # force dollar‐neutral
    dol -= dol.mean()

    # estimate today’s realized vol of that raw book
    # first compute today’s returns series per inst
    rets = np.zeros((days, insts))
    if days > 1:
        rets[1:] = (prices[1:] / prices[:-1] - 1.0)
    # simulate P&L series: pnl_t = ∑(dol * rets_t) / POS_LIMIT
    port_rets = (rets * dol).sum(axis=1) / POS_LIMIT
    hist_vol = np.std(port_rets[-VOL_LOOK:]) if days > VOL_LOOK else 1.0

    # scale to target vol
    scale = TARGET_VOL / (hist_vol or 1e-8)
    dol *= scale

    # convert to share counts
    today_price = prices[-1]
    pos = np.floor_divide(dol, today_price).astype(int)

    return pos
