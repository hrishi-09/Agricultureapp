"""Crop image scanner: OpenCV feature extraction + locally trained scikit-learn classifier.

Pipeline
  1. decode + resize, segment the leaf (non-background mask from HSV/Lab colour + largest component)
  2. extract ~50 interpretable features: colour fractions (green / yellow / brown / white / orange / dark),
     hue-saturation-value histograms, lesion blob statistics, hole fraction (pests), border scorch ratio,
     mottling (mosaic) and texture
  3. RandomForest + MLP soft-voting ensemble predicts one of the classes in data/diseases.json
  4. confidence / leaf-found guards, severity from affected leaf area, advice from the knowledge base

Training data: by default procedurally rendered leaves (synthetic_leaf) so the app works out of the box.
For real-world accuracy train on photos: `python train.py --image-dir path/` with one sub-folder per
class named like the keys in data/diseases.json (e.g. PlantVillage re-labelled to these classes).
"""
import os
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from . import load_json, MODELS

KB = load_json("diseases.json")["classes"]
CLASSES = list(KB)
PATH = os.path.join(MODELS, "vision.joblib")
SIZE = 224
W_RF = 0.7   # ensemble weight of the random forest (it generalises better than the MLP on this feature set)


# --------------------------------------------------------------------------- features
def decode(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unreadable image. Please upload a JPG or PNG photo.")
    h, w = img.shape[:2]
    s = SIZE / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def leaf_mask(img):
    """Foreground = leaf tissue (green/yellow/brown/white lesions) vs. flat background."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # plant-like hues (yellow..green..cyan-green) with some saturation OR brown/orange lesions
    plant = (h >= 15) & (h <= 95) & (s > 45) & (v > 40)
    brown = (h >= 3) & (h < 25) & (s > 60) & (v > 40)
    m = ((plant | brown) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(m.shape, np.uint8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaf = (lab == big).astype(np.uint8) * 255
    # fill interior holes (pest damage / white mildew are inside the outline) via external contours
    cnts, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(leaf)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return filled


def features(img):
    """Return (feature_vector, info_dict). info includes leaf_fraction and masks for visual overlay."""
    mask = leaf_mask(img)
    total = img.shape[0] * img.shape[1]
    leaf_area = int((mask > 0).sum())
    info = dict(leaf_fraction=leaf_area / total)
    if leaf_area < 0.04 * total:
        return np.zeros(len(FEATURE_NAMES)), dict(info, mask=mask, lesion=np.zeros_like(mask))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = (c.astype(np.float32) for c in cv2.split(hsv))
    px = mask > 0
    h, s, v = H[px], S[px], V[px]
    n = float(px.sum())
    green = (h >= 35) & (h <= 85) & (s > 60) & (v > 50)
    yellow = (h >= 20) & (h < 35) & (s > 60) & (v > 90)
    brown = (h >= 3) & (h < 20) & (s > 40) & (v < 200) & (v > 25)
    orange = (h >= 5) & (h < 22) & (s > 140) & (v > 150)
    white = (s < 55) & (v > 175)
    dark = v < 55
    fr = [x.sum() / n for x in (green, yellow, brown, orange, white, dark)]
    # lesion pixels = anything in the leaf outline that is not healthy green (incl. holes)
    healthy = ((H >= 35) & (H <= 85) & (S > 60) & (V > 50))
    lesion = (mask > 0) & ~healthy
    lesion_u8 = lesion.astype(np.uint8) * 255
    lesion_u8 = cv2.morphologyEx(lesion_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(lesion_u8)
    areas = st[1:, cv2.CC_STAT_AREA].astype(float) if nl > 1 else np.array([0.0])
    areas = areas[areas >= 6]
    nblob = len(areas)
    blob = [min(nblob, 200) / 100, (areas.mean() / n) if nblob else 0, (areas.max() / n) if nblob else 0,
            (areas.sum() / n) if nblob else 0, (areas.std() / n) if nblob > 1 else 0]
    # holes: background pixels enclosed in the outline (filled mask vs raw plant mask)
    raw = ((H >= 15) & (H <= 95) & (S > 45) & (V > 40)) | ((H >= 3) & (H < 25) & (S > 60) & (V > 40))
    holes = ((mask > 0) & ~raw & ((S < 55) | (V < 40) | (V > 215))).sum() / n
    # border scorch: non-healthy share in outer ring vs. interior
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ring = (dist > 0) & (dist <= max(6, 0.06 * np.sqrt(n)))
    inner = dist > max(6, 0.06 * np.sqrt(n))
    ring_bad = (lesion & ring).sum() / max(ring.sum(), 1)
    inner_bad = (lesion & inner).sum() / max(inner.sum(), 1)
    # mottling / colour texture inside the healthy+yellow area
    g8 = cv2.GaussianBlur(V, (0, 0), 1.5)
    bs = 16
    cells = []
    for y in range(0, img.shape[0] - bs, bs):
        for x in range(0, img.shape[1] - bs, bs):
            sub = mask[y:y + bs, x:x + bs]
            if sub.mean() > 220:
                cells.append((H[y:y + bs, x:x + bs].mean(), g8[y:y + bs, x:x + bs].mean()))
    cells = np.array(cells) if len(cells) > 4 else np.zeros((5, 2))
    mott_h, mott_v = float(cells[:, 0].std()), float(cells[:, 1].std())
    lap = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F)[px].var())
    hist_h = np.histogram(h, bins=12, range=(0, 180))[0] / n
    hist_s = np.histogram(s, bins=4, range=(0, 256))[0] / n
    hist_v = np.histogram(v, bins=4, range=(0, 256))[0] / n
    vec = np.array(fr + blob + [holes, ring_bad, inner_bad, ring_bad - inner_bad, mott_h / 20, mott_v / 40, lap / 2000,
                                h.mean() / 180, h.std() / 40, s.mean() / 255, s.std() / 80, v.mean() / 255, v.std() / 80]
                   + list(hist_h) + list(hist_s) + list(hist_v), dtype=float)
    return vec, dict(info, mask=mask, lesion=lesion_u8, affected=float(lesion.sum() / n), holes=float(holes))


FEATURE_NAMES = (["f_green", "f_yellow", "f_brown", "f_orange", "f_white", "f_dark", "n_blobs", "blob_mean", "blob_max", "blob_sum", "blob_std",
                  "holes", "ring_bad", "inner_bad", "ring_minus_inner", "mott_h", "mott_v", "laplacian", "h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std"]
                 + [f"hh{i}" for i in range(12)] + [f"hs{i}" for i in range(4)] + [f"hv{i}" for i in range(4)])


# --------------------------------------------------------------------------- synthetic leaf renderer
def _noise(rng, shape, scale=14):
    n = rng.normal(0, 1, (shape[0] // scale + 2, shape[1] // scale + 2)).astype(np.float32)
    return cv2.resize(n, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def synthetic_leaf(cls, rng):
    """Render a plausible leaf photo for a class (training data when no photo dataset is supplied)."""
    S = SIZE
    bg_kind = rng.integers(0, 4)
    if bg_kind == 0:
        bg = np.full((S, S, 3), rng.integers(200, 255), np.uint8)
    elif bg_kind == 1:
        bg = np.full((S, S, 3), (rng.integers(40, 70), rng.integers(70, 100), rng.integers(100, 140)), np.uint8)
    elif bg_kind == 2:
        g = rng.integers(60, 200)
        bg = np.full((S, S, 3), (g, g, g), np.uint8)
    else:
        bg = np.full((S, S, 3), (rng.integers(150, 200), rng.integers(170, 215), rng.integers(180, 225)), np.uint8)
    bg = np.clip(bg.astype(np.float32) + _noise(rng, (S, S), 20)[..., None] * 8, 0, 255).astype(np.uint8)
    # leaf outline (ellipse-like polygon) with random size, rotation
    cx, cy = S / 2 + rng.normal(0, 8), S / 2 + rng.normal(0, 8)
    a, b = rng.uniform(.34, .46) * S, rng.uniform(.18, .30) * S
    ang = rng.uniform(0, 180)
    t = np.linspace(0, 2 * np.pi, 80)
    pts = np.stack([a * np.cos(t) * (1 + .12 * np.cos(t)), b * np.sin(t) * (1 - .05 * np.cos(t))], 1)
    rot = np.deg2rad(ang)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    pts = (pts @ R.T + [cx, cy]).astype(np.int32)
    mask = np.zeros((S, S), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    # base colour (HSV in OpenCV units)
    hue = rng.uniform(46, 62)
    sat = rng.uniform(120, 200)
    val = rng.uniform(95, 160)
    if cls == "nitrogen_deficiency":
        hue, sat, val = rng.uniform(26, 33), rng.uniform(110, 170), rng.uniform(150, 215)
    hsv = np.zeros((S, S, 3), np.float32)
    n1 = _noise(rng, (S, S), 12)
    hsv[..., 0] = hue + n1 * 1.5
    hsv[..., 1] = sat + n1 * 8
    hsv[..., 2] = val + _noise(rng, (S, S), 6) * 6
    # veins
    vein = np.zeros((S, S), np.uint8)
    p0 = np.array([cx, cy]) - R @ [a, 0]
    p1 = np.array([cx, cy]) + R @ [a * .95, 0]
    cv2.line(vein, tuple(p0.astype(int)), tuple(p1.astype(int)), 255, 2)
    for f in np.linspace(.15, .85, 8):
        base = p0 + (p1 - p0) * f
        for sgn in (-1, 1):
            tip = base + R @ [a * .18, sgn * b * .75 * np.sin(np.pi * f)]
            cv2.line(vein, tuple(base.astype(int)), tuple(tip.astype(int)), 255, 1)
    hsv[..., 2] += (cv2.GaussianBlur(vein, (0, 0), 1) > 40) * 28
    hsv[..., 1] -= (cv2.GaussianBlur(vein, (0, 0), 1) > 40) * 25
    inside = mask > 0
    ys, xs = np.nonzero(inside)

    def blob(cx_, cy_, r, color_hsv, alpha=1.0, irregular=0.4):
        m = np.zeros((S, S), np.float32)
        k = rng.integers(7, 12)
        th = np.linspace(0, 2 * np.pi, k, endpoint=False)
        rr = r * (1 + irregular * rng.uniform(-1, 1, k))
        poly = np.stack([cx_ + rr * np.cos(th), cy_ + rr * np.sin(th)], 1).astype(np.int32)
        cv2.fillPoly(m, [poly], 1.0)
        m = cv2.GaussianBlur(m, (0, 0), max(r * .12, .6)) * alpha * (inside)
        for ch, val_ in enumerate(color_hsv):
            hsv[..., ch] = hsv[..., ch] * (1 - m) + val_ * m
        return m

    def pick():
        i = rng.integers(len(xs))
        return float(xs[i]), float(ys[i])

    holes = []
    if cls == "leaf_spot":
        for _ in range(rng.integers(10, 55)):
            x, y = pick(); r = rng.uniform(2, 6)
            blob(x, y, r * 1.8, (rng.uniform(22, 30), 150, 190), .6, .2)
            blob(x, y, r, (rng.uniform(5, 14), rng.uniform(120, 190), rng.uniform(50, 110)), 1, .25)
    elif cls == "leaf_blight":
        for _ in range(rng.integers(1, 5)):
            edge = rng.random() < .75
            if edge:
                i = rng.integers(len(pts)); x, y = pts[i]
            else:
                x, y = pick()
            blob(x, y, rng.uniform(22, 55), (rng.uniform(8, 18), rng.uniform(90, 170), rng.uniform(90, 160)), 1, .6)
    elif cls == "powdery_mildew":
        for _ in range(rng.integers(3, 12)):
            x, y = pick()
            blob(x, y, rng.uniform(12, 38), (rng.uniform(30, 60), rng.uniform(5, 40), rng.uniform(215, 252)), rng.uniform(.65, .95), .6)
    elif cls == "rust":
        for _ in range(rng.integers(40, 150)):
            x, y = pick()
            blob(x, y, rng.uniform(1.5, 3.5), (rng.uniform(8, 16), rng.uniform(170, 235), rng.uniform(170, 230)), 1, .2)
    elif cls == "potassium_deficiency":
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        w = rng.uniform(.09, .2) * min(a, b) * 2
        edge = np.clip(1 - dist / w, 0, 1) * inside
        edge = np.clip(edge + _noise(rng, (S, S), 8) * .15, 0, 1) * inside
        yel = np.clip(edge * 1.6, 0, 1)
        brn = np.clip((edge - .45) * 2.2, 0, 1)
        for ch, v1, v2 in ((0, 27, 11), (1, 150, 160), (2, 190, 105)):
            hsv[..., ch] = hsv[..., ch] * (1 - yel) + v1 * yel
            hsv[..., ch] = hsv[..., ch] * (1 - brn) + v2 * brn
    elif cls == "pest_damage":
        for _ in range(rng.integers(3, 16)):
            x, y = pick(); r = rng.uniform(3, 14)
            m = blob(x, y, r, (rng.uniform(8, 14), 120, 80), .7, .5)     # brown rim
            holes.append((x, y, r * .75))
        for _ in range(rng.integers(0, 3)):                               # chewed margin
            i = rng.integers(len(pts)); holes.append((pts[i][0], pts[i][1], rng.uniform(8, 16)))
    elif cls == "mosaic_virus":
        for _ in range(rng.integers(25, 70)):
            x, y = pick()
            blob(x, y, rng.uniform(6, 20), (rng.uniform(30, 40), rng.uniform(80, 130), rng.uniform(170, 220)), rng.uniform(.6, 1), .8)
        for _ in range(rng.integers(15, 40)):
            x, y = pick()
            blob(x, y, rng.uniform(5, 14), (rng.uniform(58, 66), rng.uniform(170, 220), rng.uniform(60, 90)), rng.uniform(.6, 1), .8)
    hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    leaf = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out = bg.copy()
    inside3 = np.repeat(mask[..., None] > 0, 3, 2)
    out[inside3] = leaf[inside3]
    for x, y, r in holes:   # holes show background
        hm = np.zeros((S, S), np.uint8)
        cv2.ellipse(hm, (int(x), int(y)), (max(int(r), 2), max(int(r * rng.uniform(.5, 1)), 2)), rng.uniform(0, 180), 0, 360, 255, -1)
        hm = (hm > 0) & (mask > 0)
        out[hm] = bg[hm]
    # photo-like degradations
    out = out.astype(np.float32)
    out *= rng.uniform(.78, 1.18)
    out += rng.normal(0, rng.uniform(1, 6), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < .5:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def synth_dataset(n_per_class=260, seed=21):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in CLASSES:
        for _ in range(n_per_class):
            vec, _ = features(synthetic_leaf(cls, rng))
            X.append(vec)
            y.append(cls)
    return np.array(X), np.array(y)


def image_dir_dataset(root, max_per_class=800):
    X, y = [], []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if cls not in KB or not os.path.isdir(d):
            continue
        for i, f in enumerate(sorted(os.listdir(d))):
            if i >= max_per_class:
                break
            try:
                img = decode(open(os.path.join(d, f), "rb").read())
            except Exception:
                continue
            vec, _ = features(img)
            X.append(vec)
            y.append(cls)
    return np.array(X), np.array(y)


# --------------------------------------------------------------------------- train / predict
def train(X=None, y=None, source="synthetic procedural leaves"):
    if X is None:
        X, y = synth_dataset()
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=.2, random_state=4, stratify=y)
    rf = RandomForestClassifier(400, min_samples_leaf=2, random_state=4, n_jobs=-1)
    nn = make_pipeline(StandardScaler(), MLPClassifier((96, 48), max_iter=500, alpha=1e-3, random_state=4))
    rf.fit(Xa, ya); nn.fit(Xa, ya)
    pe = W_RF * rf.predict_proba(Xb) + (1 - W_RF) * nn.predict_proba(Xb)
    cls = list(rf.classes_)
    pred = np.array(cls)[pe.argmax(1)]
    metrics = dict(rf_acc=float(accuracy_score(yb, rf.predict(Xb))), nn_acc=float(accuracy_score(yb, nn.predict(Xb))),
                   ens_acc=float(accuracy_score(yb, pred)), n_train=int(len(Xa)), n_test=int(len(Xb)), source=source,
                   confusion=confusion_matrix(yb, pred, labels=cls).tolist(), classes=cls)
    rf.fit(X, y); nn.fit(X, y)
    m = dict(rf=rf, nn=nn, classes=list(rf.classes_), metrics=metrics)
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(m, PATH)
    return m


def load():
    return joblib.load(PATH) if os.path.exists(PATH) else train()


def scan(m, data):
    """Analyse an uploaded photo. Returns dict with status ok / no_leaf, predictions, severity, overlay (BGR jpg bytes)."""
    img = decode(data)
    vec, info = features(img)
    thumb = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    if info["leaf_fraction"] < .04:
        return dict(status="no_leaf", message="No leaf or plant tissue could be found. Retake the photo with the affected leaf filling most of the frame, in daylight, against a plain background.", thumb=thumb)
    p = W_RF * m["rf"].predict_proba([vec])[0] + (1 - W_RF) * m["nn"].predict_proba([vec])[0]
    order = np.argsort(-p)
    top = [dict(key=m["classes"][i], name=KB[m["classes"][i]]["name"], prob=round(float(p[i]) * 100, 1)) for i in order[:3]]
    label, conf = top[0]["key"], top[0]["prob"]
    affected = info["affected"] * 100
    kb = KB[label]
    if label == "healthy":
        sev = "None"
    else:
        score = kb["severity_base"] + (2 if affected > 30 else 1 if affected > 10 else 0)
        sev = "Severe" if score >= 4 else "Moderate" if score >= 2 else "Mild"
    ov = img.copy()
    cnts, _ = cv2.findContours(info["lesion"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, [c for c in cnts if cv2.contourArea(c) >= 6], -1, (0, 0, 255), 1)
    ocnt, _ = cv2.findContours(info["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, ocnt, -1, (0, 220, 0), 1)
    overlay = cv2.imencode(".jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    return dict(status="ok", label=label, name=kb["name"], category=kb["category"], confidence=conf, top=top, severity=sev,
                affected_pct=round(affected, 1), uncertain=bool(conf < 55 or ent > .65), overlay=overlay, thumb=thumb,
                evidence=_evidence(vec, info, label))


def _evidence(vec, info, label):
    f = dict(zip(FEATURE_NAMES, vec))
    ev = [f"{info['affected'] * 100:.0f}% of the leaf area is not healthy green"]
    if f["f_yellow"] > .25 and label in ("nitrogen_deficiency", "potassium_deficiency", "mosaic_virus", "leaf_blight"):
        ev.append(f"{f['f_yellow'] * 100:.0f}% of the leaf is yellow")
    if f["f_white"] > .08 and label == "powdery_mildew":
        ev.append(f"{f['f_white'] * 100:.0f}% is white / powdery")
    if f["f_orange"] > .02:
        ev.append(f"orange-rust coloured pixels ({f['f_orange'] * 100:.1f}%)")
    if f["n_blobs"] * 100 >= 8:
        ev.append(f"about {int(f['n_blobs'] * 100)} separate lesions detected")
    if f["holes"] > .01:
        ev.append(f"{f['holes'] * 100:.1f}% of the outline is holes or missing tissue")
    if f["ring_minus_inner"] > .25:
        ev.append("damage is concentrated along the leaf edge")
    return ev


def advise(res, crop_name=None, soil=None):
    """Cultivation advice for a scan result, optionally tied to the user's soil report."""
    if res["status"] != "ok":
        return []
    kb = KB[res["label"]]
    out = [kb["advice"]]
    if res["label"] == "nitrogen_deficiency" and soil and soil.get("N") is not None:
        out.append(f"Your latest soil report shows available N = {soil['N']:g} kg/ha, which is consistent with this." if soil["N"] < 280 else
                   f"Your soil report shows adequate nitrogen ({soil['N']:g} kg/ha), so check for waterlogging, root damage or poor uptake rather than just adding more urea.")
    if res["label"] == "potassium_deficiency" and soil and soil.get("K") is not None:
        out.append(f"Soil potassium is {soil['K']:g} kg/ha ({'low' if soil['K'] < 110 else 'not low'}) in your latest report.")
    if kb["category"] in ("disease",) and soil and soil.get("ph") is not None and soil["ph"] > 7.8 and res["label"] != "healthy":
        out.append("Alkaline soil can weaken plants; correct pH and micronutrients to improve recovery.")
    if res["uncertain"]:
        out.append("The model is not very sure about this image. Retake the photo in daylight, closer, with one leaf filling the frame, or ask a local agriculture officer to confirm before spraying.")
    return out
