"""Dashboard insights: cross-feature reasoning over a user's latest records (rule-based)."""


def build(ctx):
    """ctx: latest soil/forecast/irrigation/fert/scan/crop_rec dicts (any may be None). Returns list of (severity, text)."""
    out = []
    soil, fc, irr, scan, rec = ctx.get("soil"), ctx.get("forecast"), ctx.get("irrigation"), ctx.get("scan"), ctx.get("crop_rec")
    if soil:
        if soil["grade"] == "Poor":
            out.append(("high", f"Soil health is poor ({soil['index']}/100). Fix pH and organic matter first; fertiliser will respond better afterwards."))
        elif soil["grade"] == "Moderate":
            out.append(("medium", f"Soil health is moderate ({soil['index']}/100): " + (next((t for s, t in soil["findings"] if s != "low"), "monitor each season."))))
        else:
            out.append(("low", f"Soil health is good ({soil['index']}/100)."))
    if fc:
        s = fc["summary"]
        out.append(("medium" if s["total_rain"] >= 60 or "Heat" in " ".join(s["notes"]) else "low", s["notes"][0]))
    if irr:
        out.append(("high" if irr["status"] == "Irrigate now" else "low", f"Irrigation: {irr['status']}. {irr['advice']}"))
    if scan and scan["label"] != "healthy":
        out.append(("high" if scan["severity"] == "Severe" else "medium", f"Last crop scan: {scan['name']} ({scan['severity'].lower()}). See Crop Scanner for treatment."))
    if scan and fc and scan["label"] in ("leaf_blight", "leaf_spot", "powdery_mildew", "rust") and fc["summary"]["total_rain"] > 25:
        out.append(("high", "Wet weather is forecast and your last scan found a fungal disease: apply a protective fungicide in the next dry window."))
    if rec:
        out.append(("low", f"Best crop for your latest inputs: {rec['top'][0]['name']} ({rec['top'][0]['score']:.0f}/100)."))
    if soil and irr and soil["moisture"]["available_fraction"] < .3 and irr["status"] != "Irrigate now":
        out.append(("medium", "Your last soil test showed dry soil; re-check moisture and refresh the irrigation plan."))
    if not out:
        out.append(("low", "No data yet. Start with the Farm Profile, then run Soil, Weather and Irrigation."))
    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda t: order[t[0]])
    return out
