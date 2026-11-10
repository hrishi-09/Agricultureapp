"""Fertilizer and soil-correction engine (rule-based, soil-test driven).

Crop dose (N, P2O5, K2O kg/ha) from data/crops.json is adjusted by the soil-test rating
(Low +25%, Medium 100%, High reduced) with nitrogen credits, converted to products
(DAP -> urea -> MOP) and scheduled by growth stage. pH / salinity amendments are computed
from the soil's lime requirement.
"""
from datetime import timedelta
from . import load_json

CROPS = load_json("crops.json")["crops"]
SOILS = load_json("soils.json")
TYPES, LIMITS = SOILS["types"], SOILS["limits"]
FERT = load_json("fertilizers.json")["products"]
FACTOR = {"N": {"Low": 1.25, "Medium": 1.0, "High": .75}, "P": {"Low": 1.25, "Medium": 1.0, "High": .5}, "K": {"Low": 1.25, "Medium": 1.0, "High": .5}}


def _rate(v, L):
    return "Low" if v < L["low"] else ("High" if v > L["high"] else "Medium")


def recommend(crop, soil, x, area_ha=1.0, sow_date=None):
    """x: ph, N, P, K (kg/ha available), oc (%), optional ec (dS/m), zn (ppm DTPA)."""
    c, t = CROPS[crop], TYPES[soil]
    base = dict(zip("NPK", c["fert"]))
    ratings = {k: _rate(x[k], LIMITS[k]) for k in "NPK"}
    dose = {k: base[k] * FACTOR[k][ratings[k]] for k in "NPK"}
    reasons = [f"{LIMITS[k]['label']} is {ratings[k].lower()} so the {k} dose is {int(FACTOR[k][ratings[k]] * 100)}% of the crop's standard {base[k]} kg/ha." for k in "NPK" if base[k] > 0]
    if x["oc"] >= LIMITS["oc"]["high"]:
        dose["N"] *= .9
        reasons.append("High organic carbon gives a 10% nitrogen credit.")
    dose = {k: round(v) for k, v in dose.items()}
    # product conversion
    products = []
    dap = dose["P"] / (FERT["dap"]["P"] / 100) if dose["P"] > 0 else 0
    n_from_dap = dap * FERT["dap"]["N"] / 100
    n_left = max(dose["N"] - n_from_dap, 0)
    urea = n_left / (FERT["urea"]["N"] / 100)
    mop = dose["K"] / (FERT["mop"]["K"] / 100) if dose["K"] > 0 else 0
    for key, kg in (("dap", dap), ("urea", urea), ("mop", mop)):
        if kg > 0.5:
            products.append(dict(key=key, name=FERT[key]["name"], kg_ha=round(kg), kg_total=round(kg * area_ha), bags_50kg=round(kg * area_ha / 50, 1)))
    # schedule
    s0, s1, s2 = c["n_split"]
    ini, dev, mid, late = c["stages"]
    when = [("Basal (at sowing / planting)", 0), ("Vegetative stage", ini + dev // 2), ("Flowering / grain-filling", ini + dev + mid // 3)]
    schedule = []
    for (label, day), share in zip(when, (s0, s1, s2)):
        if share <= 0:
            continue
        item = dict(stage=label, day=day, date=(sow_date + timedelta(days=day)).strftime("%d %b %Y") if sow_date else None, items=[])
        if label.startswith("Basal"):
            for key, kg in (("dap", dap), ("mop", mop)):
                if kg > .5:
                    item["items"].append(f"{FERT[key]['name']}: {kg * area_ha:.0f} kg")
            u = max(n_left * s0 / (FERT['urea']['N'] / 100) if s0 < 1 else urea, 0)
            if u > .5:
                item["items"].append(f"Urea: {u * area_ha:.0f} kg")
        else:
            u = n_left * share / (FERT["urea"]["N"] / 100)
            if u > .5:
                item["items"].append(f"Urea: {u * area_ha:.0f} kg (irrigate lightly after applying)")
        if item["items"]:
            schedule.append(item)
    # amendments
    corrections = []
    ph, opt_lo, opt_hi = x["ph"], c["ph"][1], c["ph"][2]
    if ph < opt_lo:
        target = min(max(opt_lo, 6.0), 6.5)
        lime = max((target - ph) * t["lime_per_ph"], 0)
        corrections.append(dict(kind="Lime", text=f"Soil pH {ph:.1f} is below the {c['name']} optimum ({opt_lo}-{opt_hi}). Apply about {lime:.1f} t/ha agricultural lime (target pH {target}), mixed into the top 15 cm 2-4 weeks before sowing. Split above 2 t/ha across two seasons.", amount_t_ha=round(lime, 1)))
    elif ph > 8.5:
        corrections.append(dict(kind="Gypsum", text=f"pH {ph:.1f} suggests sodic soil. Apply gypsum at 2.5-5 t/ha, then leach with good-quality water; confirm with an ESP/EC test.", amount_t_ha=3.5))
    elif ph > opt_hi + .3:
        sulphur = (ph - opt_hi) * 400 * (1.5 if t["infil"] < 8 else 1)
        corrections.append(dict(kind="Acidifying", text=f"pH {ph:.1f} is above the optimum ({opt_lo}-{opt_hi}). Add elemental sulphur at about {sulphur:.0f} kg/ha (indicative), plus 5-10 t/ha compost; prefer ammonium sulphate over urea.", amount_t_ha=round(sulphur / 1000, 2)))
    if x.get("ec") is not None and x["ec"] >= 4:
        corrections.append(dict(kind="Salinity", text=f"EC {x['ec']:g} dS/m is high. Provide drainage and leach with 1.5x the normal irrigation before sowing; choose salt-tolerant varieties.", amount_t_ha=None))
    if x["oc"] < LIMITS["oc"]["low"]:
        corrections.append(dict(kind="Organic matter", text="Organic carbon is low. Add 10 t/ha FYM or 5 t/ha compost before sowing and grow a green-manure crop in the off-season.", amount_t_ha=10))
    zn = x.get("zn")
    if zn is not None:
        if zn < .6:
            corrections.append(dict(kind="Zinc", text="DTPA zinc is below 0.6 ppm. Apply 25 kg/ha zinc sulphate basal (once in 3 seasons) or 0.5% foliar spray.", amount_t_ha=.025))
    elif ph > 7.5 or t["infil"] >= 30 or crop in ("rice", "maize"):
        corrections.append(dict(kind="Zinc (check)", text="Zinc is commonly short on this soil/crop. Get a DTPA zinc test; if it is below 0.6 ppm apply 25 kg/ha zinc sulphate.", amount_t_ha=None))
    if ratings["P"] == "Low" and ph < 5.5:
        reasons.append("Low pH also reduces phosphorus availability, so liming will raise the efficiency of the P applied.")
    tips = ["Apply fertiliser on moist soil and irrigate lightly afterwards; avoid applying before heavy rain.",
            "Place phosphorus and potash near the root zone (band or drill) rather than broadcasting."]
    if crop in ("soybean", "chickpea", "groundnut"):
        tips.append("Treat seed with Rhizobium culture; the legume fixes its own nitrogen so heavy urea is not needed.")
    if t["infil"] >= 30:
        tips.append("Sandy soil leaches nutrients: use smaller, more frequent N splits or neem-coated urea.")
    return dict(crop=c["name"], dose=dose, ratings=ratings, products=products, schedule=schedule, corrections=corrections, reasons=reasons, tips=tips, area_ha=area_ha)
