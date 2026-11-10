"""Crop recommendation.

Training data is generated from the agronomic requirement ranges in data/crops.json
(FAO EcoCrop-style trapezoids) with realistic measurement noise, or loaded from a real
CSV (columns: N,P,K,ph,temp,humidity,rain,soil,label) via train.py --crop-csv.
A RandomForest + GradientBoosting soft-voting ensemble is trained locally. The final
score blends the ML probability with the rule-based suitability so results stay
explainable: every recommendation lists which factors fit and which limit the crop.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from . import load_json, trap, MODELS

CROPS = load_json("crops.json")["crops"]
SOILS = load_json("soils.json")["types"]
SOIL_KEYS = list(SOILS)
FEATS = ["N", "P", "K", "ph", "temp", "humidity", "rain"] + [f"soil_{s}" for s in SOIL_KEYS]
PATH = os.path.join(MODELS, "crop.joblib")


def _sample_in(rng, lo_abs, lo_opt, hi_opt, hi_abs):
    """Sample mostly from the optimum with some tails towards the absolute limits."""
    u = rng.random()
    if u < .7:
        return rng.uniform(lo_opt, hi_opt)
    if u < .85:
        return rng.uniform(lo_abs, lo_opt)
    return rng.uniform(hi_opt, hi_abs)


def synth(n_per_crop=700, seed=11):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for key, c in CROPS.items():
        soils = list(c["soils"])
        w = np.array([c["soils"][s] for s in soils]) ** 2
        w = w / w.sum()
        hi_n, hi_p, hi_k = c["soil_npk"][0]
        lo_n, lo_p, lo_k = c["soil_npk"][1]
        for _ in range(n_per_crop):
            rows.append(dict(
                N=np.clip(rng.normal(hi_n * .75, 25), 5, 250), P=np.clip(rng.normal(hi_p * .75, 12), 2, 150),
                K=np.clip(rng.normal(hi_k * .75, 18), 5, 250), ph=_sample_in(rng, *c["ph"]),
                temp=_sample_in(rng, *c["temp"]), humidity=_sample_in(rng, *c["humidity"]),
                rain=_sample_in(rng, *c["rain"]), soil=rng.choice(soils, p=w)))
            labels.append(key)
    df = pd.DataFrame(rows)
    # sensor noise
    df["temp"] += rng.normal(0, 1.0, len(df))
    df["humidity"] = (df["humidity"] + rng.normal(0, 3, len(df))).clip(5, 100)
    df["rain"] = (df["rain"] * rng.normal(1, .06, len(df))).clip(50)
    return df, np.array(labels)


def featurize(df):
    X = df[["N", "P", "K", "ph", "temp", "humidity", "rain"]].astype(float).copy()
    for s in SOIL_KEYS:
        X[f"soil_{s}"] = (df["soil"] == s).astype(float)
    return X[FEATS]


def train(df=None, y=None):
    if df is None:
        df, y = synth()
    X = featurize(df)
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=.2, random_state=3, stratify=y)
    rf = RandomForestClassifier(120, min_samples_leaf=8, max_depth=16, random_state=3, n_jobs=-1).fit(Xa, ya)
    gb = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=3).fit(Xa, ya)
    proba = (rf.predict_proba(Xb) + gb.predict_proba(Xb)) / 2
    classes = rf.classes_
    top3 = float(np.mean([yt in classes[np.argsort(-pr)[:3]] for yt, pr in zip(yb, proba)]))
    metrics = dict(top3_acc=top3, rf_acc=float(accuracy_score(yb, rf.predict(Xb))), gb_acc=float(accuracy_score(yb, gb.predict(Xb))),
                   ens_acc=float(accuracy_score(yb, classes[proba.argmax(1)])), n_train=int(len(Xa)), n_test=int(len(Xb)))
    # Re-fit on all data for the shipped model
    rf.fit(X, y)
    gb.fit(X, y)
    m = dict(rf=rf, gb=gb, classes=list(rf.classes_), metrics=metrics, feats=FEATS)
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(m, PATH)
    return m


def load():
    return joblib.load(PATH) if os.path.exists(PATH) else train()


def rule_score(c, x):
    """Per-factor suitability in [0,1] from the crop's requirement trapezoids."""
    f = dict(
        temperature=trap(x["temp"], *c["temp"]), rainfall=trap(x["rain"], *c["rain"]),
        humidity=trap(x["humidity"], *c["humidity"]), ph=trap(x["ph"], *c["ph"]),
        soil_type=float(c["soils"].get(x["soil"], 0.3)))
    hi, lo = c["soil_npk"]
    npk = []
    for v, h, l in zip((x["N"], x["P"], x["K"]), hi, lo):
        npk.append(1.0 if v >= h * .7 else max(0.25, .25 + .75 * (v - l * .5) / max(h * .7 - l * .5, 1)))
    f["nutrients"] = float(np.mean(npk))
    # geometric mean punishes a single hard limiting factor
    vals = np.array(list(f.values())) + 1e-3
    return float(min(np.exp(np.log(vals).mean()), 1.0)), f


def recommend(m, x, top=5):
    """x: dict N,P,K,ph,temp,humidity,rain,soil. Returns ranked crops with reasons."""
    df = pd.DataFrame([x])
    X = featurize(df)
    p = (m["rf"].predict_proba(X)[0] + m["gb"].predict_proba(X)[0]) / 2
    ml = dict(zip(m["classes"], p))
    out = []
    for key, c in CROPS.items():
        rs, factors = rule_score(c, x)
        score = .5 * ml.get(key, 0) / (max(p.max(), 1e-9)) + .5 * rs  # ML relative to best + rule suitability
        good = [k for k, v in factors.items() if v >= .85]
        limit = sorted([(k, v) for k, v in factors.items() if v < .6], key=lambda t: t[1])
        out.append(dict(key=key, name=c["name"], season=c["season"], score=round(score * 100, 1), ml_prob=round(float(ml.get(key, 0)) * 100, 1),
                        suitability=round(rs * 100, 1), factors={k: round(v * 100) for k, v in factors.items()},
                        good=good, limiting=[k for k, _ in limit], expected_yield_t_ha=c["yield_t"]))
    out.sort(key=lambda r: -r["score"])
    return out[:top]


def explain(rec, x):
    c = CROPS[rec["key"]]
    bits = []
    if rec["good"]:
        bits.append("well matched on " + ", ".join(g.replace("_", " ") for g in rec["good"]))
    if rec["limiting"]:
        notes = []
        for l in rec["limiting"]:
            if l == "ph":
                notes.append(f"soil pH {x['ph']:.1f} is outside the ideal {c['ph'][1]}-{c['ph'][2]}")
            elif l == "temperature":
                notes.append(f"temperature {x['temp']:.0f} C is outside the ideal {c['temp'][1]}-{c['temp'][2]} C")
            elif l == "rainfall":
                notes.append(f"rainfall {x['rain']:.0f} mm vs ideal {c['rain'][1]}-{c['rain'][2]} mm")
            elif l == "humidity":
                notes.append(f"humidity {x['humidity']:.0f}% vs ideal {c['humidity'][1]}-{c['humidity'][2]}%")
            elif l == "soil_type":
                notes.append(f"{x['soil']} soil is not ideal")
            elif l == "nutrients":
                notes.append("soil N-P-K is below what this crop prefers")
        bits.append("limited by: " + "; ".join(notes))
    return ". ".join(bits).capitalize() + "." if bits else "Moderately suitable."
