"""Soil analysis: Soil Health Card style ratings and a weighted health index (rule-based, transparent)."""
from . import load_json, trap

SOILS = load_json("soils.json")
TYPES, LIMITS = SOILS["types"], SOILS["limits"]


def rate(value, low, high):
    return "Low" if value < low else ("High" if value > high else "Medium")


def score_level(v, low, high):
    """0-100: below 'low' scales 0-60, low..high 60-90, above 'high' 90-100."""
    if v <= low:
        return 60 * max(v, 0) / low
    if v <= high:
        return 60 + 30 * (v - low) / (high - low)
    return min(100, 90 + 10 * (v - high) / high)


def analyse(x):
    """x: dict ph, moisture (vol %), N, P, K (kg/ha), oc (%), ec (dS/m, optional), soil (type key)."""
    t = TYPES[x["soil"]]
    out = {"nutrients": {}, "scores": {}, "findings": []}
    ph = x["ph"]
    out["ph_class"] = ("Strongly acidic" if ph < 5.5 else "Moderately acidic" if ph < 6.0 else "Slightly acidic" if ph < 6.5 else
                       "Neutral" if ph <= 7.5 else "Slightly alkaline" if ph <= 8.0 else "Moderately alkaline" if ph <= 8.5 else "Strongly alkaline / sodic risk")
    out["scores"]["pH"] = round(100 * trap(ph, 4.0, 6.0, 7.5, 9.5))
    for k in ("N", "P", "K", "oc"):
        L = LIMITS[k]
        r = rate(x[k], L["low"], L["high"])
        out["nutrients"][k] = dict(label=L["label"], value=x[k], unit=L["unit"], rating=r, low=L["low"], high=L["high"])
        out["scores"][k if k != "oc" else "Organic C"] = round(score_level(x[k], L["low"], L["high"]))
    ec = x.get("ec")
    if ec is not None:
        out["ec_class"] = "Normal" if ec < 1 else "Slightly saline" if ec < 2 else "Moderately saline" if ec < 4 else "Highly saline"
        out["scores"]["Salinity"] = round(100 * trap(ec, -1, 0, 1, 6))
    # moisture status relative to this soil's retention
    fc, wp = t["fc"] * 100, t["wp"] * 100
    aw = (x["moisture"] - wp) / (fc - wp)
    out["moisture"] = dict(value=x["moisture"], fc=round(fc, 1), wp=round(wp, 1), available_fraction=round(aw, 2),
                           status="Waterlogged / saturated" if x["moisture"] > fc * 1.1 else "Adequate (near field capacity)" if aw >= .75 else
                           "Moderate" if aw >= .45 else "Dry - irrigation due soon" if aw >= .15 else "Very dry - at or near wilting point")
    weights = {"pH": .20, "N": .17, "P": .17, "K": .17, "Organic C": .20, "Salinity": .09}
    used = {k: w for k, w in weights.items() if k in out["scores"]}
    idx = sum(out["scores"][k] * w for k, w in used.items()) / sum(used.values())
    out["index"] = round(idx)
    out["grade"] = "Good" if idx >= 75 else "Moderate" if idx >= 55 else "Poor"
    out["soil_name"], out["soil_note"] = t["name"], t["notes"]
    f = out["findings"]
    if ph < 5.5:
        f.append(("high", f"Soil is strongly acidic (pH {ph:.1f}): phosphorus, calcium and magnesium are less available and aluminium may be toxic. Apply lime (see Fertilizer page for the dose)."))
    elif ph < 6.0:
        f.append(("medium", f"Soil is moderately acidic (pH {ph:.1f}). Light liming improves nutrient availability for most crops."))
    elif ph > 8.5:
        f.append(("high", f"Soil is strongly alkaline (pH {ph:.1f}), indicating sodic risk. Apply gypsum, add organic matter and leach with good-quality water."))
    elif ph > 7.8:
        f.append(("medium", f"Soil is alkaline (pH {ph:.1f}): zinc, iron and phosphorus may be locked up. Use organic matter and acid-forming fertilisers (ammonium sulphate); consider zinc sulphate."))
    for k, n in out["nutrients"].items():
        if n["rating"] == "Low":
            sev = "high" if n["value"] < n["low"] * .5 else "medium"
            f.append((sev, f"{n['label']} is low ({n['value']:g} {n['unit']}, limit {n['low']:g}). " + {
                "N": "Plan split nitrogen doses and add compost or a legume in rotation.",
                "P": "Apply DAP/SSP as a basal dose placed near the seed.",
                "K": "Apply muriate or sulphate of potash.",
                "oc": "Add 5-10 t/ha FYM or compost and retain crop residue."}[k]))
        elif n["rating"] == "High" and k != "oc":
            f.append(("low", f"{n['label']} is high ({n['value']:g} {n['unit']}): you can reduce or skip this nutrient in the fertiliser plan."))
    if ec is not None and ec >= 2:
        f.append(("high" if ec >= 4 else "medium", f"Salinity is {out['ec_class'].lower()} (EC {ec:g} dS/m). Improve drainage, leach salts and choose tolerant crops."))
    m = out["moisture"]
    if m["available_fraction"] < .3:
        f.append(("medium", "Soil is dry for this soil type: see the Irrigation page for a schedule."))
    elif x["moisture"] > fc * 1.1:
        f.append(("medium", "Soil is saturated. Improve drainage and delay fertiliser and heavy machinery until it drains."))
    if not f:
        f.append(("low", "All measured parameters are within a healthy range. Maintain organic matter and monitor each season."))
    order = {"high": 0, "medium": 1, "low": 2}
    f.sort(key=lambda t: order[t[0]])
    return out
