"""Smart irrigation: FAO-56 style water balance.

ETo  : Hargreaves-Samani from forecast Tmax/Tmin and extraterrestrial radiation (latitude + day of year).
ETc  : ETo x Kc, where Kc follows the crop's growth stage (days after sowing).
Root zone: TAW = 1000 (FC - WP) Zr ; irrigate when depletion reaches MAD x TAW.
Rain : USDA-SCS effective rainfall.  Rice adds puddled-field percolation losses.
"""
import math
from datetime import date, datetime, timedelta
from . import load_json

CROPS = load_json("crops.json")["crops"]
SOILS = load_json("soils.json")["types"]
EFF = {"flood": .55, "sprinkler": .75, "drip": .90}


def ra_mm(lat_deg, doy):
    """Extraterrestrial radiation expressed as mm/day of evaporation equivalent."""
    phi = math.radians(lat_deg)
    dr = 1 + .033 * math.cos(2 * math.pi / 365 * doy)
    dec = .409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    ws = math.acos(max(-1, min(1, -math.tan(phi) * math.tan(dec))))
    ra = 24 * 60 / math.pi * .0820 * dr * (ws * math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.sin(ws))
    return .408 * ra


def eto(tmax, tmin, lat, doy):
    tm = (tmax + tmin) / 2
    return max(0.0, .0023 * (tm + 17.8) * math.sqrt(max(tmax - tmin, 0)) * ra_mm(lat, doy))


def eff_rain(mm):
    """USDA-SCS effective rainfall per daily event."""
    return 0.0 if mm < 2.5 else max(0.0, (mm * (125 - .2 * mm) / 125) if mm <= 250 / 3 else 125 / 3 + .1 * mm) * .9


def kc_at(c, das):
    ini, dev, mid, late = c["stages"]
    k0, k1, k2 = c["kc"]
    if das < 0:
        return k0, "pre-sowing"
    if das < ini:
        return k0, "initial"
    if das < ini + dev:
        return k0 + (k1 - k0) * (das - ini) / dev, "development"
    if das < ini + dev + mid:
        return k1, "mid-season"
    if das < ini + dev + mid + late:
        return k1 + (k2 - k1) * (das - ini - dev - mid) / late, "late season"
    return k2, "beyond season"


def plan(crop, soil, moisture, sow_date, forecast, lat, method="flood", area_ha=1.0, today=None):
    c, s = CROPS[crop], SOILS[soil]
    today = today or date.today()
    notes = []
    if sow_date is None:
        sow_date = today - timedelta(days=30)
        notes.append("No sowing date given: assumed the crop is 30 days old.")
    das0 = (today - sow_date).days
    taw = 1000 * (s["fc"] - s["wp"]) * c["root"]
    raw = c["mad"] * taw
    theta = max(min(moisture / 100, s["fc"] * 1.15), 0)
    dr = min(max(1000 * (s["fc"] - theta) * c["root"], 0), taw)      # current root-zone depletion (mm)
    eff = EFF.get(method, .55)
    perc = min(max(s["infil"] * .25, 1), 8) if crop == "rice" else 0.0
    rows, events, dr_sim = [], [], dr
    for i, d in enumerate(forecast):
        dt = datetime.strptime(d["date"], "%Y-%m-%d").date()
        doy = dt.timetuple().tm_yday
        e0 = eto(d["tmax"], d["tmin"], lat, doy)
        kc, stage = kc_at(c, das0 + i)
        etc = e0 * kc + perc
        pe = eff_rain(d["rain_mm"] / max(d["rain_prob"] / 100, .05)) * d["rain_prob"] / 100 if d["rain_prob"] >= 35 else 0.0
        dr_sim = min(max(dr_sim + etc - pe, 0), taw * 1.0)
        irr_net = 0.0
        if dr_sim >= raw:
            irr_net = dr_sim                     # refill to field capacity
            dr_sim = 0.0
            events.append(dict(date=d["date"], net_mm=round(irr_net, 1), gross_mm=round(irr_net / eff, 1)))
        rows.append(dict(date=d["date"], eto=round(e0, 2), kc=round(kc, 2), etc=round(etc, 2), rain=round(pe, 1), depletion=round(dr_sim, 1), irrigate_mm=round(irr_net / eff, 1)))
    total_gross = sum(e["gross_mm"] for e in events)
    dr0_frac = dr / taw if taw else 0
    if moisture > s["fc"] * 100 * 1.1 and crop != "rice":
        status, advice = "Do not irrigate", "Soil is above field capacity. Irrigating now risks waterlogging; improve drainage."
    elif dr >= raw:
        gross = dr / eff
        status, advice = "Irrigate now", f"Root-zone water is at or below the allowed depletion. Apply about {gross:.0f} mm ({gross * 10:,.0f} m3/ha)."
    elif events:
        e = events[0]
        status, advice = f"Irrigate on {e['date']}", f"Soil water is adequate today; next irrigation of about {e['gross_mm']:.0f} mm expected on {e['date']}."
    else:
        status, advice = "No irrigation needed this week", "Soil water plus forecast rainfall covers crop demand for the next 7 days."
    _, stage_now = kc_at(c, das0)
    if stage_now == "beyond season":
        notes.append("The crop is past its normal season length; harvest or confirm the sowing date.")
    if stage_now in ("mid-season",) and crop in ("wheat", "maize", "cotton", "soybean", "chickpea"):
        notes.append("Mid-season is the critical stage for yield: avoid water stress during flowering and grain/pod filling.")
    if method == "flood" and crop not in ("rice", "sugarcane", "jute"):
        notes.append("Drip or sprinkler irrigation could cut this requirement by 25-40% compared with flood.")
    if s["infil"] >= 30:
        notes.append("Sandy soil: split each irrigation into smaller, more frequent applications to avoid deep percolation.")
    if s["infil"] <= 5:
        notes.append("Low infiltration soil: apply water slowly or in cycles to avoid runoff and ponding.")
    return dict(status=status, advice=advice, stage=stage_now, das=das0, taw=round(taw, 1), raw=round(raw, 1), depletion_now=round(dr, 1),
                depletion_pct=round(100 * dr0_frac), efficiency=eff, method=method, events=events, rows=rows,
                week_gross_mm=round(total_gross, 1), week_litres=round(total_gross * 10000 * area_ha),
                week_etc=round(sum(r["etc"] for r in rows), 1), week_rain=round(sum(r["rain"] for r in rows), 1), notes=notes, area_ha=area_ha)
