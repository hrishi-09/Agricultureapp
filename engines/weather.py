"""Weather prediction.

Models (all trained locally on a multi-year daily series):
  * temp_max, temp_min, humidity, pressure : GradientBoostingRegressor on lag features
  * rain_prob                               : GradientBoostingClassifier (rain >= 2.5 mm)
  * rain_mm                                 : GradientBoostingRegressor trained on rainy days

Training series: either a user-supplied daily CSV (date,tmax,tmin,rh,rain,pres) via
train.py --weather-csv, or a stochastic daily generator driven by the regional climate
normals in data/regions.json (seasonal cycle + AR(1) anomalies + monsoon rain state
machine). Prediction for a user takes their last observed days (manual entry or saved
history) and rolls the model forward day by day for 7 days.
"""
import os
import zlib
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
from . import load_json, MODELS

REGIONS = load_json("regions.json")["regions"]
PATH = os.path.join(MODELS, "weather.joblib")
LAGS = 3
RAIN_T = 2.5


def _interp(arr, doy):
    """Smooth seasonal curve from 12 monthly values (periodic cubic-ish via cosine interpolation)."""
    mid = np.array([15.5 + 30.4 * i for i in range(12)])
    a = np.array(arr, float)
    xs = np.concatenate([mid - 365, mid, mid + 365])
    ys = np.concatenate([a, a, a])
    return np.interp(doy, xs, ys)


def generate(region, years=8, seed=5):
    """Generate a daily series from regional normals."""
    r = REGIONS[region]
    rng = np.random.default_rng(seed + zlib.crc32(region.encode()) % 1000)
    days = pd.date_range("2016-01-01", periods=int(365 * years), freq="D")
    doy = days.dayofyear.values
    tmax_c, tmin_c = _interp(r["tmax"], doy), _interp(r["tmin"], doy)
    rh_c, pres_c = _interp(r["rh"], doy), _interp(r["pres"], doy)
    rain_month = _interp(r["rain"], doy)           # mm / month
    rdays = np.clip(_interp(r["rdays"], doy), .05, 28)
    p_rain = np.clip(rdays / 30.4, .01, .85)
    mean_wet = np.maximum(rain_month / (rdays + 1e-6), 3.0)
    n = len(days)
    a_t = a_h = a_p = 0.0
    wet = False
    tmax, tmin, rh, pres, rain = (np.zeros(n) for _ in range(5))
    for i in range(n):
        # Markov rain occurrence (persistence)
        p = np.clip(p_rain[i] * (1.5 if wet else .75), 0, .95)
        wet = rng.random() < p
        amt = rng.gamma(.8, mean_wet[i] / .8) if wet else 0.0
        rain[i] = amt
        cloud = 1.0 if wet else 0.0
        a_t = .7 * a_t + rng.normal(0, 1.1)
        a_h = .75 * a_h + rng.normal(0, 4.0)
        a_p = .85 * a_p + rng.normal(0, 1.2)
        tmax[i] = tmax_c[i] + a_t - 2.2 * cloud
        tmin[i] = tmin_c[i] + .5 * a_t + .6 * cloud + rng.normal(0, .5)
        rh[i] = np.clip(rh_c[i] + a_h + 10 * cloud, 8, 100)
        pres[i] = pres_c[i] + a_p - 2.0 * cloud
    return pd.DataFrame(dict(date=days, tmax=tmax, tmin=np.minimum(tmin, tmax - 2), rh=rh, rain=rain, pres=pres))


def _season(doy):
    ang = 2 * np.pi * (doy - 1) / 365
    return np.sin(ang), np.cos(ang)


def _row(hist, doy, lat):
    """Features from the last LAGS observed days (hist: list of dict newest last)."""
    h = hist[-LAGS:]
    s, c = _season(doy)
    f = [s, c, lat]
    for k in ("tmax", "tmin", "rh", "pres", "rain"):
        f += [d[k] for d in reversed(h)]
    f += [np.mean([d["rain"] for d in hist[-7:]]), np.mean([d["tmax"] for d in hist[-7:]]) ]
    return f


def _frame(df, lat):
    recs = df.to_dict("records")
    X, Y = [], []
    for i in range(7, len(recs) - 1):
        X.append(_row(recs[:i + 1], recs[i + 1]["date"].dayofyear, lat))
        Y.append(recs[i + 1])
    return np.array(X), pd.DataFrame(Y)


def train(region_csv=None, regions=None):
    regions = regions or list(REGIONS)
    Xs, Ys, tag = [], [], []
    for rg in regions:
        df = region_csv[rg] if region_csv and rg in region_csv else generate(rg)
        X, Y = _frame(df, REGIONS[rg]["lat"])
        Xs.append(X)
        Ys.append(Y)
    X = np.vstack(Xs)
    Y = pd.concat(Ys, ignore_index=True)
    split = int(len(X) * .85)
    # chronological split inside each region is approximated by contiguous blocks; keep simple & shuffled-free
    idx = np.arange(len(X))
    test_mask = (idx % 100) >= 85
    tr, te = ~test_mask, test_mask
    models, metrics = {}, {}
    for tgt in ("tmax", "tmin", "rh", "pres"):
        g = GradientBoostingRegressor(n_estimators=140, max_depth=3, learning_rate=.08, random_state=2).fit(X[tr], Y[tgt][tr])
        metrics[f"{tgt}_mae"] = float(mean_absolute_error(Y[tgt][te], g.predict(X[te])))
        models[tgt] = g.fit(X, Y[tgt])
    yb = (Y["rain"] >= RAIN_T).astype(int).values
    clf = GradientBoostingClassifier(n_estimators=140, max_depth=3, learning_rate=.08, random_state=2).fit(X[tr], yb[tr])
    pr = clf.predict_proba(X[te])[:, 1]
    metrics["rain_acc"] = float(accuracy_score(yb[te], pr > .5))
    metrics["rain_auc"] = float(roc_auc_score(yb[te], pr))
    metrics["rain_base_rate"] = float(yb.mean())
    models["rain_p"] = clf.fit(X, yb)
    wet = yb == 1
    reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=.08, random_state=2).fit(X[tr & wet], Y["rain"][tr & wet])
    metrics["rain_mm_mae_wet"] = float(mean_absolute_error(Y["rain"][te & wet], reg.predict(X[te & wet])))
    models["rain_mm"] = reg.fit(X[wet], Y["rain"][wet])
    models["metrics"] = metrics
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(models, PATH)
    return models


def load():
    return joblib.load(PATH) if os.path.exists(PATH) else train()


def default_history(region, today):
    """When the user has not entered recent observations, use the climatology for the date."""
    r = REGIONS[region]
    out = []
    for k in range(7, 0, -1):
        d = today - pd.Timedelta(days=k)
        doy = d.dayofyear
        rain_m = float(_interp(r["rain"], doy)); rd = max(float(_interp(r["rdays"], doy)), .05)
        out.append(dict(tmax=float(_interp(r["tmax"], doy)), tmin=float(_interp(r["tmin"], doy)), rh=float(_interp(r["rh"], doy)),
                        pres=float(_interp(r["pres"], doy)), rain=rain_m / 30.4))
    return out


def forecast(models, region, hist, start, days=7):
    """hist: list of >=7 daily dicts (tmax,tmin,rh,pres,rain) oldest->newest. Returns daily forecast."""
    lat = REGIONS[region]["lat"]
    hist = [dict(d) for d in hist[-7:]]
    out = []
    for k in range(days):
        date = start + pd.Timedelta(days=k)
        X = np.array([_row(hist, date.dayofyear, lat)])
        tmax = float(models["tmax"].predict(X)[0])
        tmin = float(min(models["tmin"].predict(X)[0], tmax - 2))
        rh = float(np.clip(models["rh"].predict(X)[0], 5, 100))
        pres = float(models["pres"].predict(X)[0])
        p = float(models["rain_p"].predict_proba(X)[0, 1])
        mm = float(max(models["rain_mm"].predict(X)[0], 0)) * p   # expected rainfall
        out.append(dict(date=date.strftime("%Y-%m-%d"), tmax=round(tmax, 1), tmin=round(tmin, 1), rh=round(rh), pres=round(pres, 1),
                        rain_prob=round(p * 100), rain_mm=round(mm, 1)))
        hist.append(dict(tmax=tmax, tmin=tmin, rh=rh, pres=pres, rain=mm))
    return out


def summarize(fc):
    """Plain-language agronomic outlook derived from the numeric forecast."""
    rain = sum(d["rain_mm"] for d in fc)
    notes = []
    if max(d["tmax"] for d in fc) >= 40:
        notes.append("Heat stress risk: afternoon temperatures may reach 40 C or more. Irrigate in the evening and avoid fertiliser top-dressing.")
    if min(d["tmin"] for d in fc) <= 4:
        notes.append("Frost/cold risk: night temperature may fall to 4 C or below; protect nurseries and sensitive crops.")
    if rain >= 60:
        notes.append(f"Heavy rain expected ({rain:.0f} mm over {len(fc)} days): postpone spraying and fertiliser, ensure field drainage.")
    elif rain < 5:
        notes.append("Little or no rain expected: plan irrigation.")
    if np.mean([d["rh"] for d in fc]) >= 80 and np.mean([d["tmax"] for d in fc]) >= 22:
        notes.append("Warm and humid: fungal diseases (blight, rust, leaf spot) are likely; scout fields and consider preventive sprays.")
    if not notes:
        notes.append("Stable conditions expected, suitable for normal field operations.")
    return dict(total_rain=round(rain, 1), notes=notes)


def season_climate(region, today, months=4):
    """Climatological mean temperature, humidity and total rainfall for the next `months` months (for crop advice defaults)."""
    r = REGIONS[region]
    idx = [(today.month - 1 + i) % 12 for i in range(months)]
    return dict(temp=round(float(np.mean([(r["tmax"][i] + r["tmin"][i]) / 2 for i in idx])), 1),
                humidity=round(float(np.mean([r["rh"][i] for i in idx]))),
                rain=round(float(sum(r["rain"][i] for i in idx))))
