"""Optional local LLM (Ollama) for the chatbot. Talks only to YOUR Ollama server (default http://127.0.0.1:11434).

The LLM is grounded (retrieval-augmented): every answer is built from
  * the farmer's own data (soil report, forecast, irrigation plan, crop ranking, fertiliser plan, last scan),
  * numbers computed by the rule engines (so doses/schedules are never invented by the model),
  * passages retrieved from data/knowledge.json,
and the built-in NLP answer is passed in as "verified answer". If Ollama is off or fails, the app falls back to the
built-in NLP chatbot automatically. Uses only the standard library (no extra dependencies).

Env: OLLAMA_URL, OLLAMA_MODEL (default llama3.2:3b), OLLAMA_ENABLED (0 to disable), OLLAMA_TIMEOUT (seconds).
"""
import json
import os
import time
import urllib.request

URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
ENABLED = os.getenv("OLLAMA_ENABLED", "1") != "0"
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
_state = {"t": 0.0, "ok": False, "why": "not checked"}

SYSTEM = """You are AgriXAI, a practical farming assistant for Indian farmers. You run locally on the farmer's own server.
Rules:
- Answer the farmer's question using ONLY the FARM DATA, VERIFIED ANSWER and KNOWLEDGE sections below.
- Never invent numbers, fertiliser doses, pesticide names, dates or prices. Use the numbers given. If something needed is missing, say which page of the app to use (Soil, Weather, Irrigation, Crop Advisor, Fertilizer, Crop Scanner).
- For crop recommendation, name the best crop(s) from the ranking and say why; explain what must be improved in soil (pH, nutrients, organic matter) and in cultivation (sowing time, water, fertiliser, pest/disease care).
- Use simple words and short numbered steps. At most 180 words. Reply in the same language as the question.
- For pesticides, remind the farmer to read the label and follow local agriculture-officer advice.
"""


def status(force=False):
    """(ok, message). Cached for a few seconds so an absent Ollama never slows the app."""
    if not ENABLED:
        return False, "disabled (OLLAMA_ENABLED=0)"
    now = time.time()
    if not force and now - _state["t"] < (30 if _state["ok"] else 10):
        return _state["ok"], _state["why"]
    ok, why = False, ""
    try:
        with urllib.request.urlopen(URL + "/api/tags", timeout=1.5) as r:
            names = [m.get("name", "") for m in json.load(r).get("models", [])]
        if MODEL in names or MODEL + ":latest" in names:
            ok, why = True, MODEL
        else:
            why = f"model {MODEL} not pulled (run: ollama pull {MODEL})"
    except Exception:
        why = f"Ollama not reachable at {URL}"
    _state.update(t=now, ok=ok, why=why)
    return ok, why


def facts(ctx, extra=None):
    """Compact text description of the farmer's situation for the prompt."""
    L = []
    f = ctx.get("farm") or {}
    if f:
        L.append(f"Farm: region {f.get('region')}, soil type {f.get('soil')}, area {f.get('area_ha')} ha, current crop {f.get('crop') or 'none'}, sowing date {f.get('sow_date') or 'not set'}, irrigation {f.get('method')}.")
    s = ctx.get("soil")
    if s:
        L.append(f"Soil report ({s['when']}): {s['soil_name']}, pH {s['ph']:.1f} ({s['ph_class']}), moisture {s['moisture']['value']}% ({s['moisture']['status']}), health index {s['index']}/100 ({s['grade']}). "
                 + "; ".join(f"{n['label']} {n['value']:g} {n['unit']} = {n['rating']}" for n in s["nutrients"].values()) + ".")
        L.append("Soil problems: " + " ".join(t for sev, t in s["findings"][:4]))
    r = ctx.get("crop_rec")
    if r:
        L.append("Crop ranking (score/100, with reasons): " + " | ".join(f"{c['name']} {c['score']:.0f}: {c.get('why', '')}" for c in r["top"][:4]))
    fc = ctx.get("forecast")
    if fc:
        d = fc["days"]
        L.append(f"Weather next {len(d)} days ({fc['region_name']}): max {max(x['tmax'] for x in d):.0f} C, min {min(x['tmin'] for x in d):.0f} C, rain {fc['summary']['total_rain']:.0f} mm. " + " ".join(fc["summary"]["notes"]))
    ir = ctx.get("irrigation")
    if ir:
        L.append(f"Irrigation ({ir['crop']}, {ir['stage']} stage): {ir['status']}. {ir['advice']} Next 7 days: irrigate {ir['week_gross_mm']} mm in total.")
    sc = ctx.get("scan")
    if sc:
        L.append(f"Latest crop scan: {sc['name']} ({sc['severity']}, {sc['confidence']:.0f}% confidence, {sc['affected_pct']:.0f}% leaf affected).")
    if extra and extra.get("plan"):
        p = extra["plan"]
        L.append(f"Fertiliser plan for {p['crop']} (computed from the soil report): N {p['dose']['N']}, P2O5 {p['dose']['P']}, K2O {p['dose']['K']} kg/ha. Products: "
                 + ", ".join(f"{x['name']} {x['kg_ha']} kg/ha" for x in p["products"]) + ". Corrections: " + (" ".join(c["text"] for c in p["corrections"]) or "none") + ".")
    return "\n".join(L) if L else "No data saved yet for this farmer."


def build_messages(question, ctx, verified, passages, history, extra=None):
    know = "\n".join(f"- {p['title']}: {p['text']}" for p in passages) or "- (none found)"
    system = (SYSTEM + "\nFARM DATA:\n" + facts(ctx, extra) + "\n\nVERIFIED ANSWER (from the app's rule engines; keep its numbers):\n" + (verified or "(none)") + "\n\nKNOWLEDGE:\n" + know)
    msgs = [{"role": "system", "content": system}]
    for role, text in history[-6:]:
        msgs.append({"role": role, "content": text[:800]})
    msgs.append({"role": "user", "content": question})
    return msgs


def stream(messages):
    """Yield text chunks from Ollama /api/chat (streaming). Raises on connection/HTTP errors."""
    body = json.dumps({"model": MODEL, "messages": messages, "stream": True, "keep_alive": "30m",
                       "options": {"temperature": 0.3, "num_ctx": 4096, "num_predict": 420}}).encode()
    req = urllib.request.Request(URL + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        for line in r:
            if not line.strip():
                continue
            j = json.loads(line)
            if j.get("error"):
                raise RuntimeError(j["error"])
            t = (j.get("message") or {}).get("content", "")
            if t:
                yield t
            if j.get("done"):
                return
