"""Local NLP chatbot (no external AI).

  * Intent classification : TF-IDF (word 1-2grams + char 2-5grams) -> LogisticRegression, trained on data/intents.json
  * Entity extraction     : crop names (aliases), disease classes, soil terms, numbers
  * Retrieval             : TF-IDF cosine similarity over the agronomy knowledge base (data/knowledge.json)
  * Personalisation       : answers are composed from the user's own latest soil report, forecast, irrigation plan,
                            fertiliser plan, crop scan and history passed in as `ctx`; nothing is canned.
  * Dialogue memory       : last crop / topic kept by the caller and passed back in.
"""
import os
import re
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from . import load_json, MODELS
from .crop import CROPS
from . import fertilizer
from .vision import KB as DISEASES

PATH = os.path.join(MODELS, "chatbot.joblib")
INTENTS = load_json("intents.json")["intents"]
PASSAGES = load_json("knowledge.json")["passages"]

LABEL = {"N": "nitrogen", "P": "phosphorus", "K": "potassium"}
DISEASE_WORDS = {"leaf_blight": ["blight"], "leaf_spot": ["leaf spot", "spots", "spot"], "powdery_mildew": ["mildew", "powdery"], "rust": ["rust"],
                 "nitrogen_deficiency": ["nitrogen deficiency", "n deficiency", "yellowing", "yellow leaves"], "potassium_deficiency": ["potassium deficiency", "potash deficiency", "scorch"],
                 "pest_damage": ["pest", "insect", "caterpillar", "holes"], "mosaic_virus": ["mosaic", "virus"]}


def _vec():
    return (TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True))


def train():
    texts, labels = [], []
    for it, ex in INTENTS.items():
        for e in ex:
            texts.append(e); labels.append(it)
    # light augmentation: crop-name substitution so intents stay stable when a crop is mentioned
    for it in ("irrigation", "fertilizer", "crop_info", "season_planning", "disease_treatment", "pest_control", "yield"):
        for k, c in CROPS.items():
            for e in INTENTS[it][:4]:
                texts.append(f"{e} for {c['aliases'][0]}"); labels.append(it)
    w, ch = _vec()
    X = hstack([w.fit_transform(texts), ch.fit_transform(texts)]).tocsr()
    clf = LogisticRegression(C=12, max_iter=2000)
    cv = float(cross_val_score(clf, X, labels, cv=4).mean())
    clf.fit(X, labels)
    pw = [f"{p['title']} {p['keywords']} {p['text']}" for p in PASSAGES]
    rv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    R = rv.fit_transform(pw)
    m = dict(w=w, ch=ch, clf=clf, rv=rv, R=R, metrics=dict(intent_cv_acc=cv, n_utterances=len(texts), n_intents=len(INTENTS), n_passages=len(PASSAGES)))
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(m, PATH)
    return m


def load():
    return joblib.load(PATH) if os.path.exists(PATH) else train()


def entities(text):
    t = " " + re.sub(r"[^a-z0-9. ]", " ", text.lower()) + " "
    crop = None
    for k, c in CROPS.items():
        if any(f" {a} " in t for a in c["aliases"] + [k]):
            crop = k
            break
    dis = None
    for k, words in DISEASE_WORDS.items():
        if any(w in t for w in words):
            dis = k
            break
    return dict(crop=crop, disease=dis)


def classify(m, text):
    X = hstack([m["w"].transform([text]), m["ch"].transform([text])]).tocsr()
    p = m["clf"].predict_proba(X)[0]
    i = int(p.argmax())
    return m["clf"].classes_[i], float(p[i]), dict(zip(m["clf"].classes_, p))


def retrieve(m, text, crop=None, k=2):
    q = m["rv"].transform([text])
    sims = cosine_similarity(q, m["R"])[0]
    if crop:
        for i, p in enumerate(PASSAGES):
            if p["crop"] == crop:
                sims[i] *= 1.6
    order = np.argsort(-sims)[:k]
    return [(PASSAGES[i], float(sims[i])) for i in order if sims[i] > .08]


def _cn(k):
    return CROPS[k]["name"]


def respond(m, text, ctx, mem):
    """ctx: dict of the user's latest records (soil, forecast, irrigation, fert, scan, crop_rec, history, farm).
    mem: dict persisted by caller (last_crop). Returns (reply, intent, confidence, new_mem)."""
    text = text.strip()
    intent, conf, _ = classify(m, text)
    ent = entities(text)
    crop = ent["crop"] or mem.get("crop") or (ctx.get("farm") or {}).get("crop")
    mem = dict(mem)
    if ent["crop"]:
        mem["crop"] = ent["crop"]
    lines = []
    farm = ctx.get("farm") or {}
    soil, fc, irr, fert, scan, rec = ctx.get("soil"), ctx.get("forecast"), ctx.get("irrigation"), ctx.get("fert"), ctx.get("scan"), ctx.get("crop_rec")
    # strong lexical overrides when the classifier is unsure
    if conf < .35:
        intent = "knowledge"
    if ent["disease"] and intent in ("knowledge", "pest_control", "scan_result") and intent != "scan_result":
        intent = "disease_treatment"

    def need(what, page):
        return f"I don't have a {what} yet. Open the **{page}** page and run it, then ask me again and I'll use your numbers."

    if intent == "greeting":
        lines.append(f"Hello{', ' + ctx['name'] if ctx.get('name') else ''}! I'm your local farm assistant. I answer from your own soil reports, weather forecasts, irrigation and fertiliser plans and crop scans. What would you like to know?")
    elif intent == "thanks":
        lines.append("You're welcome! Ask me anything else about your fields.")
    elif intent == "goodbye":
        lines.append("Goodbye, and good luck with the crop!")
    elif intent == "help":
        lines.append("I can: (1) explain your latest soil report, (2) summarise the weather forecast, (3) tell you when and how much to irrigate, (4) suggest crops and fertiliser doses, (5) explain a crop-image scan and how to treat it, (6) answer general farming questions from my agronomy library, and (7) summarise your farming history.")
    elif intent == "my_soil":
        if not soil:
            lines.append(need("soil report", "Soil"))
        else:
            lines.append(f"Your latest soil report ({soil['when']}) gives a health index of **{soil['index']}/100 ({soil['grade']})** for {soil['soil_name']} soil with pH {soil['ph']:.1f} ({soil['ph_class'].lower()}).")
            lines.append("Nutrients: " + ", ".join(f"{n['label'].lower()} {n['rating'].lower()} ({n['value']:g})" for n in soil["nutrients"].values()) + ".")
            top = [f for f in soil["findings"] if f[0] != "low"][:3] or soil["findings"][:1]
            lines.append("Main points: " + " ".join(f"{i + 1}) {t}" for i, (_, t) in enumerate(top)))
    elif intent == "my_weather":
        if not fc:
            lines.append(need("weather forecast", "Weather"))
        else:
            d = fc["days"]
            lines.append(f"For {fc['region']}, the next {len(d)} days: temperature {min(x['tmin'] for x in d):.0f}-{max(x['tmax'] for x in d):.0f} C, humidity about {np.mean([x['rh'] for x in d]):.0f}%, total expected rain {fc['summary']['total_rain']:.0f} mm.")
            wet = max(d, key=lambda x: x["rain_prob"])
            lines.append(f"Highest rain chance is {wet['rain_prob']}% on {wet['date']}. " + " ".join(fc["summary"]["notes"]))
    elif intent == "irrigation":
        if not irr:
            lines.append(need("irrigation plan", "Irrigation"))
        else:
            lines.append(f"**{irr['status']}.** {irr['advice']}")
            lines.append(f"{irr['crop']} is in the {irr['stage']} stage (day {irr['das']}); the root zone is {irr['depletion_pct']}% depleted. Over the next 7 days the crop will use about {irr['week_etc']:.0f} mm, rain should supply {irr['week_rain']:.0f} mm, and I'd schedule {irr['week_gross_mm']:.0f} mm of irrigation ({irr['week_litres']:,} litres for {irr['area_ha']:g} ha, {irr['method']}).")
    elif intent == "crop_recommend":
        if not rec:
            lines.append(need("crop recommendation", "Crop Advisor"))
        else:
            r = rec["top"]
            lines.append(f"Based on your latest inputs ({rec['when']}), the best match is **{r[0]['name']}** (score {r[0]['score']:.0f}/100), followed by {r[1]['name']} ({r[1]['score']:.0f}) and {r[2]['name']} ({r[2]['score']:.0f}).")
            lines.append(rec["explanation"])
    elif intent in ("fertilizer", "ph_fix"):
        plan = None
        if soil and crop:   # compute a plan live from the user's own soil report
            x = dict(ph=soil["ph"], N=soil["N"], P=soil["P"], K=soil["K"], oc=soil["oc"], ec=soil.get("ec"))
            plan = fertilizer.recommend(crop, soil["soil"], x, farm.get("area_ha") or 1.0)
        if intent == "ph_fix":
            if soil:
                lines.append(f"Your soil pH is {soil['ph']:.1f} ({soil['ph_class'].lower()}).")
            ph_items = [c for c in (plan or {}).get("corrections", []) if c["kind"] in ("Lime", "Gypsum", "Acidifying", "Salinity")]
            if ph_items:
                lines.append(" ".join(c["text"] for c in ph_items))
            elif soil and 6.0 <= soil["ph"] <= 7.5:
                lines.append("That is in the range most crops like, so no pH correction is needed.")
            elif not soil:
                lines.append(need("soil report", "Soil"))
            ks = retrieve(m, text, None, 1)
            if ks:
                lines.append(ks[0][0]["text"])
        elif plan:
            lines.append(f"For {plan['crop']} on your soil ({', '.join(f'{LABEL[k]} {v.lower()}' for k, v in plan['ratings'].items())}): N {plan['dose']['N']}, P2O5 {plan['dose']['P']}, K2O {plan['dose']['K']} kg/ha.")
            if plan["products"]:
                lines.append("In products: " + ", ".join(f"{p['name']} {p['kg_ha']} kg/ha" for p in plan["products"]) + ".")
            if plan["corrections"]:
                lines.append("Also: " + " ".join(c["text"] for c in plan["corrections"][:2]))
            lines.append("Open the **Fertilizer** page for the full split schedule.")
        elif soil:
            low = [f"{n['label'].lower()} is {n['rating'].lower()}" for n in soil["nutrients"].values() if n["rating"] != "Medium"]
            lines.append(("Your soil report shows " + "; ".join(low) + ". ") if low else "Your soil nutrients are in the medium range. ")
            lines.append("Tell me the crop (for example 'fertilizer for wheat') and I'll calculate doses from your soil report.")
        else:
            lines.append(need("soil report", "Soil"))
    elif intent == "scan_result":
        if not scan:
            lines.append(need("crop scan", "Crop Scanner"))
        else:
            lines.append(f"Your latest scan ({scan['when']}) shows **{scan['name']}** with {scan['confidence']:.0f}% confidence, severity {scan['severity'].lower()} (about {scan['affected_pct']:.0f}% of the leaf affected).")
            kb = DISEASES[scan["label"]]
            lines.append("Treatment: " + " ".join(kb["treatment"][:2]))
            lines.append("Prevention: " + kb["prevention"][0])
            if scan.get("uncertain"):
                lines.append("The model was not fully certain, so please confirm with a local expert before spraying.")
    elif intent in ("disease_treatment", "pest_control"):
        key = ent["disease"] or (scan["label"] if scan and scan["label"] != "healthy" and intent == "disease_treatment" and not ent["disease"] else None)
        if intent == "pest_control" and not ent["disease"]:
            key = "pest_damage"
        if key:
            kb = DISEASES[key]
            lines.append(f"**{kb['name']}**: {kb['cause']}")
            lines.append("What to do: " + " ".join(kb["treatment"]))
            lines.append("Prevention: " + " ".join(kb["prevention"][:2]))
            if scan and scan["label"] == key:
                lines.append(f"This matches your latest scan ({scan['when']}).")
            if fc and key not in ("pest_damage", "nitrogen_deficiency", "potassium_deficiency") and fc["summary"]["total_rain"] > 25:
                lines.append("Heavy rain is forecast, so wait for a dry window before spraying or reapply after rain.")
        else:
            for p, s in retrieve(m, text, crop, 2):
                lines.append(p["text"])
    elif intent == "history":
        h = ctx.get("history") or {}
        if not any(h.values()):
            lines.append("You have no saved records yet. Run the soil, weather, crop and scan tools and I'll summarise them here.")
        else:
            lines.append("Your records: " + ", ".join(f"{v} {k}" for k, v in h.items() if v) + ".")
            if ctx.get("insights"):
                lines.append("Highlights: " + " ".join(ctx["insights"][:3]))
    elif intent == "yield":
        lines.append("Top yield levers, in order of impact: fix soil pH/organic matter, apply a soil-test based fertiliser dose, irrigate at critical stages, control weeds early and scout for pests weekly.")
        if soil:
            tops = [t for s_, t in soil["findings"] if s_ != "low"][:2]
            if tops:
                lines.append("For your soil specifically: " + " ".join(tops))
        if irr:
            lines.append(f"Irrigation status: {irr['status']}.")
    else:  # organic, season_planning, crop_info, knowledge, anything else -> retrieval
        res = retrieve(m, text, crop, 2)
        if res:
            lines.append(res[0][0]["text"])
            if len(res) > 1 and res[1][1] > .6 * res[0][1]:
                lines.append(res[1][0]["text"])
            if crop and fc and intent in ("season_planning", "crop_info"):
                lines.append(f"Current forecast for your region: {fc['summary']['notes'][0]}")
        else:
            lines.append("I couldn't find a confident answer in my library. Try asking about a specific crop, soil problem, irrigation, fertiliser, pests or diseases, or use the tools in the sidebar for your own data.")
            conf = 0.0
    if crop and intent in ("crop_info", "season_planning") and CROPS.get(crop):
        c = CROPS[crop]
        lines.append(f"Quick facts for {c['name']}: season {c['season']}, ideal temperature {c['temp'][1]}-{c['temp'][2]} C, soil pH {c['ph'][1]}-{c['ph'][2]}, seasonal rainfall {c['rain'][1]}-{c['rain'][2]} mm.")
    mem["intent"] = intent
    return "\n\n".join(lines), intent, round(conf, 2), mem
