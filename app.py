"""AgriXAI - AI-powered smart agriculture platform. Every model runs locally (scikit-learn + OpenCV); no external AI APIs."""
import base64
import json
import os
import re
import secrets
import time
from datetime import date, datetime, timedelta, timezone
from functools import wraps

import bcrypt
import pandas as pd
from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, session
from sqlalchemy import text

import db
from engines import chatbot, crop, fertilizer, insights, irrigation, llm, soil, vision, weather, MODELS

db.init()
M_CROP, M_WEATHER, M_VISION, M_CHAT = crop.load(), weather.load(), vision.load(), chatbot.load()
REPORT = {}
try:
    REPORT = json.load(open(os.path.join(MODELS, "report.json")))
except Exception:
    pass

app = Flask(__name__)
on_render = bool(os.getenv("RENDER"))
key = os.getenv("SECRET_KEY")
if not key:
    if on_render:
        raise RuntimeError("SECRET_KEY must be set in production")
    key = secrets.token_hex(32)           # dev: sessions reset on restart
app.secret_key = key
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=on_render,
                  MAX_CONTENT_LENGTH=6 * 1024 * 1024, PERMANENT_SESSION_LIFETIME=timedelta(days=7))
TZ = timezone(timedelta(minutes=int(os.getenv("TZ_OFFSET_MIN", "330"))))     # default IST
CROPS, SOILS, REGIONS = crop.CROPS, soil.TYPES, weather.REGIONS


def today():
    return datetime.now(TZ).date()


# ------------------------------------------------------------------ security helpers
@app.context_processor
def ctx():
    session.setdefault("csrf", secrets.token_hex(16))
    return dict(csrf=session["csrf"], user=session.get("user"), CROPS=CROPS, SOILS=SOILS, REGIONS=REGIONS)


@app.before_request
def csrf():
    if request.method == "POST":
        tok = request.form.get("csrf") or request.headers.get("X-CSRF")
        if not tok or tok != session.get("csrf"):
            abort(400)


@app.after_request
def headers(r):
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["Referrer-Policy"] = "same-origin"
    return r


def login_required(f):
    @wraps(f)
    def w(*a, **k):
        return f(*a, **k) if session.get("user") else redirect("/login")
    return w


def admin_required(f):
    @wraps(f)
    def w(*a, **k):
        return f(*a, **k) if session.get("user", {}).get("role") == "admin" else abort(403)
    return w


_attempts = {}


def throttled(ip, limit=8, window=300):
    now = time.time()
    hits = [t for t in _attempts.get(ip, []) if now - t < window]
    _attempts[ip] = hits
    return len(hits) >= limit


def num(name, lo, hi, default=None):
    raw = (request.form.get(name) or "").strip()
    if raw == "":
        if default is None:
            abort(400, f"{name} is required")
        return default
    try:
        v = float(raw)
    except ValueError:
        abort(400, f"{name} must be a number")
    if v != v or v < lo or v > hi:
        abort(400, f"{name} must be between {lo:g} and {hi:g}")
    return v


def choice(name, allowed, default=None):
    v = request.form.get(name, default)
    if v not in allowed:
        abort(400, f"invalid {name}")
    return v


def parse_date(name):
    raw = (request.form.get(name) or "").strip()
    if not raw:
        return None
    try:
        d = datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        abort(400, f"{name} must be a date")
    if d < date(2000, 1, 1) or d > today() + timedelta(days=365):
        abort(400, f"{name} out of range")
    return d


@app.errorhandler(400)
def bad(e):
    return render_template("error.html", code=400, msg=getattr(e, "description", "Bad request")), 400


@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403, msg="You do not have access to this page."), 403


@app.errorhandler(413)
def too_big(e):
    return render_template("error.html", code=413, msg="File too large (max 6 MB)."), 413


# ------------------------------------------------------------------ context helpers
def when(rec):
    return rec["created"].strftime("%d %b %Y, %H:%M") + " UTC"


def uid():
    return session["user"]["id"]


def farm():
    f = db.get_farm(uid())
    return f


def latest_ctx(u):
    """Everything the chatbot, dashboard and insights need: the user's latest results."""
    c = {"name": session["user"]["name"].split()[0], "farm": None}
    f = db.get_farm(u)
    c["farm"] = dict(region=f.region, soil=f.soil, area_ha=f.area_ha, crop=f.crop, sow_date=f.sow_date, method=f.method)
    r = db.latest(u, "soil")
    if r:
        c["soil"] = dict(r["data"], when=when(r))
    r = db.latest(u, "weather")
    if r:
        c["forecast"] = dict(r["data"], when=when(r))
    r = db.latest(u, "irrigation")
    if r:
        c["irrigation"] = dict(r["data"], when=when(r))
    r = db.latest(u, "fertilizer")
    if r:
        c["fert"] = dict(r["data"], when=when(r))
    r = db.latest(u, "scan")
    if r:
        c["scan"] = dict(r["data"], when=when(r))
    r = db.latest(u, "crop")
    if r:
        c["crop_rec"] = dict(r["data"], when=when(r))
    if "crop_rec" not in c and "soil" in c:       # no saved ranking yet: compute one from the soil report + regional climate
        sd, sc_ = c["soil"], weather.season_climate(f.region, today())
        x = dict(N=sd["N"], P=sd["P"], K=sd["K"], ph=sd["ph"], soil=sd["soil"], **sc_)
        top = crop.recommend(M_CROP, x, top=5)
        for t in top:
            t["why"] = crop.explain(t, x)
        c["crop_rec"] = dict(top=top, inputs=x, explanation=top[0]["why"], when="computed now from your soil report")
    cnt = db.counts(u)
    c["history"] = {k: cnt.get(k, 0) for k in ("soil", "weather", "irrigation", "crop", "fertilizer", "scan")}
    c["insights"] = [t for _, t in insights.build(c)]
    return c


def current_forecast(u, region, force=False, obs=None):
    """Return the stored forecast for today/region, or build one now and save it."""
    r = db.latest(u, "weather")
    t = today()
    if r and not force and r["data"].get("region") == region and r["data"].get("start") == t.strftime("%Y-%m-%d") and not obs:
        return r["data"]
    start = pd.Timestamp(t)
    hist = weather.default_history(region, start)
    if obs:
        hist[-1].update(obs)
    days = weather.forecast(M_WEATHER, region, hist, start, 7)
    data = dict(region=region, region_name=REGIONS[region]["name"], start=start.strftime("%Y-%m-%d"), days=days, summary=weather.summarize(days),
                used_obs=bool(obs))
    db.save(u, "weather", data, f"{REGIONS[region]['name']}: {min(d['tmin'] for d in days):.0f}-{max(d['tmax'] for d in days):.0f} C, rain {data['summary']['total_rain']:.0f} mm")
    return data


# ------------------------------------------------------------------ auth
@app.route("/")
def home():
    return redirect("/dashboard")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()
        if throttled(ip):
            flash("Too many attempts. Please wait a few minutes.")
            return render_template("auth.html", mode="login"), 429
        with db.Sess() as s:
            u = s.query(db.User).filter_by(email=request.form.get("email", "").strip().lower()).first()
            if u and u.active and bcrypt.checkpw(request.form.get("password", "").encode(), u.pw.encode()):
                session.clear()
                session["user"] = dict(id=u.id, name=u.name, role=u.role)
                session.permanent = True
                return redirect("/dashboard")
        _attempts.setdefault(ip, []).append(time.time())
        flash("Invalid credentials or disabled account.")
    return render_template("auth.html", mode="login")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        n, e, p, c = (request.form.get(k, "").strip() for k in ("name", "email", "password", "confirm"))
        e = e.lower()
        if not n or len(n) > 80 or not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+$", e):
            flash("Enter a name and a valid email.")
        elif len(p) < 8 or p != c:
            flash("Passwords must match and be 8+ characters.")
        else:
            with db.Sess() as s:
                if s.query(db.User).filter_by(email=e).first():
                    flash("Email already registered.")
                else:
                    s.add(db.User(name=n, email=e, pw=db.hp(p)))
                    s.commit()
                    flash("Registered. Please sign in.")
                    return redirect("/login")
    return render_template("auth.html", mode="register")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect("/login")


# ------------------------------------------------------------------ dashboard & farm
@app.route("/dashboard")
@login_required
def dashboard():
    c = latest_ctx(uid())
    return render_template("dashboard.html", c=c, ins=insights.build(c), report=REPORT)


@app.route("/farm", methods=["GET", "POST"])
@login_required
def farm_profile():
    if request.method == "POST":
        cr = request.form.get("crop") or None
        if cr and cr not in CROPS:
            abort(400, "invalid crop")
        db.update_farm(uid(), region=choice("region", REGIONS), soil=choice("soil", SOILS), area_ha=num("area_ha", .01, 10000),
                       crop=cr, sow_date=parse_date("sow_date"), method=choice("method", irrigation.EFF))
        flash("Farm profile saved.")
        return redirect("/farm")
    return render_template("farm.html", f=farm())


# ------------------------------------------------------------------ weather
@app.route("/weather", methods=["GET", "POST"])
@login_required
def weather_page():
    f = farm()
    data, region = None, f.region
    if request.method == "POST":
        region = choice("region", REGIONS)
        obs = {}
        for k, lo, hi in (("tmax", -10, 55), ("tmin", -20, 40), ("rh", 0, 100), ("pres", 500, 1100), ("rain", 0, 600)):
            v = (request.form.get(k) or "").strip()
            if v:
                obs[k] = num(k, lo, hi)
        if "tmax" in obs and "tmin" in obs and obs["tmin"] >= obs["tmax"]:
            abort(400, "Minimum temperature must be lower than maximum")
        data = current_forecast(uid(), region, force=True, obs=obs or None)
    else:
        r = db.latest(uid(), "weather")
        if r:
            data = r["data"]
            region = data.get("region", region)
    return render_template("weather.html", data=data, region=region, metrics=REPORT.get("weather", {}), hist=db.recent(uid(), "weather", 8))


# ------------------------------------------------------------------ soil
@app.route("/soil", methods=["GET", "POST"])
@login_required
def soil_page():
    f = farm()
    res = form = None
    if request.method == "POST":
        ecs = (request.form.get("ec") or "").strip()
        form = dict(soil=choice("soil", SOILS), ph=num("ph", 3, 10), moisture=num("moisture", 0, 70), N=num("N", 0, 1500), P=num("P", 0, 300),
                    K=num("K", 0, 1500), oc=num("oc", 0, 6), ec=num("ec", 0, 40) if ecs else None)
        res = soil.analyse(form)
        inputs = {k: v for k, v in form.items() if k != "moisture"}   # res["moisture"] already holds the moisture analysis
        data = dict(res, **inputs)
        db.save(uid(), "soil", data, f"{res['soil_name']} soil, pH {form['ph']:.1f}, health {res['index']}/100 ({res['grade']})")
        res = data
    else:
        r = db.latest(uid(), "soil")
        form = dict(r["data"], moisture=r["data"]["moisture"]["value"]) if r else dict(soil=f.soil, ph=6.5, moisture=25, N=250, P=15, K=150, oc=.6, ec=None)
        if r:
            res = r["data"]
    return render_template("soil.html", res=res, form=form, hist=db.recent(uid(), "soil", 8))


# ------------------------------------------------------------------ irrigation
@app.route("/irrigation", methods=["GET", "POST"])
@login_required
def irrigation_page():
    f = farm()
    res = None
    s_rec = db.latest(uid(), "soil")
    form = dict(crop=f.crop or "wheat", soil=f.soil, moisture=(s_rec["data"]["moisture"]["value"] if s_rec else 25.0),
                sow_date=f.sow_date.strftime("%Y-%m-%d") if f.sow_date else "", method=f.method, area_ha=f.area_ha, region=f.region)
    if request.method == "POST":
        form = dict(crop=choice("crop", CROPS), soil=choice("soil", SOILS), moisture=num("moisture", 0, 70), method=choice("method", irrigation.EFF),
                    area_ha=num("area_ha", .01, 10000), region=choice("region", REGIONS), sow_date=request.form.get("sow_date", ""))
        sd = parse_date("sow_date")
        fc = current_forecast(uid(), form["region"])
        res = irrigation.plan(form["crop"], form["soil"], form["moisture"], sd, fc["days"], REGIONS[form["region"]]["lat"], form["method"], form["area_ha"], today())
        res.update(crop=CROPS[form["crop"]]["name"], crop_key=form["crop"])
        db.save(uid(), "irrigation", res, f"{res['crop']}: {res['status']}")
    else:
        r = db.latest(uid(), "irrigation")
        if r:
            res = r["data"]
    return render_template("irrigation.html", res=res, form=form, hist=db.recent(uid(), "irrigation", 8))


# ------------------------------------------------------------------ crop recommendation
@app.route("/crops", methods=["GET", "POST"])
@login_required
def crops_page():
    f = farm()
    s_rec = db.latest(uid(), "soil")
    sc = weather.season_climate(f.region, today())
    sd = s_rec["data"] if s_rec else {}
    form = dict(N=sd.get("N", 90), P=sd.get("P", 40), K=sd.get("K", 40), ph=sd.get("ph", 6.5), soil=sd.get("soil", f.soil), **sc)
    res = None
    if request.method == "POST":
        form = dict(N=num("N", 0, 1500), P=num("P", 0, 300), K=num("K", 0, 1500), ph=num("ph", 3, 10), temp=num("temp", -5, 50),
                    humidity=num("humidity", 0, 100), rain=num("rain", 0, 6000), soil=choice("soil", SOILS))
        top = crop.recommend(M_CROP, form, top=6)
        for t in top:
            t["why"] = crop.explain(t, form)
        res = dict(top=top, inputs=form, explanation=top[0]["why"])
        db.save(uid(), "crop", res, f"Top: {top[0]['name']} ({top[0]['score']:.0f}), {top[1]['name']}, {top[2]['name']}")
    else:
        r = db.latest(uid(), "crop")
        if r:
            res = r["data"]
            form = res["inputs"]
    return render_template("crops.html", res=res, form=form, metrics=REPORT.get("crop", {}), hist=db.recent(uid(), "crop", 8))


# ------------------------------------------------------------------ fertilizer
@app.route("/fertilizer", methods=["GET", "POST"])
@login_required
def fertilizer_page():
    f = farm()
    s_rec = db.latest(uid(), "soil")
    sd = s_rec["data"] if s_rec else {}
    form = dict(crop=f.crop or "wheat", soil=sd.get("soil", f.soil), ph=sd.get("ph", 6.5), N=sd.get("N", 250), P=sd.get("P", 15), K=sd.get("K", 150),
                oc=sd.get("oc", .6), ec=sd.get("ec"), zn="", area_ha=f.area_ha, sow_date=f.sow_date.strftime("%Y-%m-%d") if f.sow_date else "")
    res = None
    if request.method == "POST":
        ecs, zns = (request.form.get("ec") or "").strip(), (request.form.get("zn") or "").strip()
        form = dict(crop=choice("crop", CROPS), soil=choice("soil", SOILS), ph=num("ph", 3, 10), N=num("N", 0, 1500), P=num("P", 0, 300), K=num("K", 0, 1500),
                    oc=num("oc", 0, 6), ec=num("ec", 0, 40) if ecs else None, zn=num("zn", 0, 50) if zns else "", area_ha=num("area_ha", .01, 10000),
                    sow_date=request.form.get("sow_date", ""))
        x = dict(ph=form["ph"], N=form["N"], P=form["P"], K=form["K"], oc=form["oc"], ec=form["ec"], zn=form["zn"] if zns else None)
        res = fertilizer.recommend(form["crop"], form["soil"], x, form["area_ha"], parse_date("sow_date"))
        res["crop_key"] = form["crop"]
        db.save(uid(), "fertilizer", res, f"{res['crop']}: N{res['dose']['N']} P{res['dose']['P']} K{res['dose']['K']} kg/ha")
    else:
        r = db.latest(uid(), "fertilizer")
        if r:
            res = r["data"]
    return render_template("fertilizer.html", res=res, form=form, hist=db.recent(uid(), "fertilizer", 8))


# ------------------------------------------------------------------ crop image scanner
@app.route("/scanner", methods=["GET", "POST"])
@login_required
def scanner_page():
    res = None
    if request.method == "POST":
        fobj = request.files.get("image")
        if not fobj or not fobj.filename:
            abort(400, "Choose an image to scan")
        data = fobj.read()
        try:
            r = vision.scan(M_VISION, data)
        except ValueError as e:
            abort(400, str(e))
        ck = request.form.get("crop") or None
        if ck and ck not in CROPS:
            abort(400, "invalid crop")
        c = latest_ctx(uid())
        if r["status"] == "ok":
            r["advice"] = vision.advise(r, CROPS[ck]["name"] if ck else None, c.get("soil"))
            r["kb"] = vision.KB[r["label"]]
            r["crop"] = CROPS[ck]["name"] if ck else None
            rec = {k: v for k, v in r.items() if k not in ("overlay", "thumb")}
            rid = db.save(uid(), "scan", rec, f"{r['name']} ({r['confidence']:.0f}%) - {r['severity']}", blob=r["overlay"])
            r["rid"] = rid
        res = r
    else:
        rid = request.args.get("id", type=int)
        r = None
        if rid:
            for row in db.recent(uid(), "scan", 200):
                if row["id"] == rid:
                    r = row
        else:
            r = db.latest(uid(), "scan")
        if r:
            res = dict(r["data"], status="ok", rid=r["id"], kb=vision.KB[r["data"]["label"]])
    return render_template("scanner.html", res=res, metrics=REPORT.get("vision", {}), hist=db.recent(uid(), "scan", 12))


@app.route("/scan-image/<int:rid>")
@login_required
def scan_image(rid):
    b = db.get_blob(uid(), rid)
    if not b:
        abort(404)
    return Response(b, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


# ------------------------------------------------------------------ chatbot
@app.route("/chat")
@login_required
def chat_page():
    msgs = list(reversed(db.recent(uid(), "chat", 30)))
    return render_template("chat.html", msgs=msgs, metrics=REPORT.get("chatbot", {}))


@app.route("/api/chat", methods=["POST"])
@login_required
def chat_api():
    q = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not q or len(q) > 500:
        return jsonify(error="Message must be 1-500 characters"), 400
    c = latest_ctx(uid())
    reply, intent, conf, mem = chatbot.respond(M_CHAT, q, c, session.get("chat_mem", {}))
    session["chat_mem"] = mem
    db.save(uid(), "chat", dict(q=q, a=reply, intent=intent, conf=conf), q[:200])
    return jsonify(reply=reply, intent=intent, confidence=conf)


@app.route("/api/chat/status")
@login_required
def chat_status():
    ok, why = llm.status()
    return jsonify(llm=ok, model=llm.MODEL if ok else None, detail=why, mode="ollama" if ok else "builtin")


@app.route("/api/chat/stream", methods=["POST"])
@login_required
def chat_stream():
    """Hybrid chatbot: built-in NLP + rule engines compute a verified answer; Ollama (if running) rewrites it as a grounded,
    conversational answer streamed token by token. Falls back to the built-in answer if Ollama is unavailable."""
    q = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not q or len(q) > 500:
        return jsonify(error="Message must be 1-500 characters"), 400
    u = uid()
    c = latest_ctx(u)
    reply, intent, conf, mem = chatbot.respond(M_CHAT, q, c, session.get("chat_mem", {}))
    session["chat_mem"] = mem
    ok, _ = llm.status()
    hist = []
    for r in reversed(db.recent(u, "chat", 4)):
        hist += [("user", r["data"].get("q", "")), ("assistant", r["data"].get("a", ""))]
    target = mem.get("crop") or c["farm"].get("crop") or (c["crop_rec"]["top"][0]["key"] if c.get("crop_rec") else None)
    extra = {}
    if c.get("soil") and target in CROPS:
        sd = c["soil"]
        extra["plan"] = fertilizer.recommend(target, sd["soil"], dict(ph=sd["ph"], N=sd["N"], P=sd["P"], K=sd["K"], oc=sd["oc"], ec=sd.get("ec")), c["farm"].get("area_ha") or 1.0)
    passages = [p for p, _ in chatbot.retrieve(M_CHAT, q, target, 3)]

    def ev(**k):
        return json.dumps(k) + "\n"

    def gen():
        out, used = "", "builtin"
        try:
            if ok:
                for tok in llm.stream(llm.build_messages(q, c, reply, passages, hist, extra)):
                    used = "ollama"
                    out += tok
                    yield ev(t=tok)
            if not out:
                out = reply
                yield ev(t=reply)
        except Exception:
            llm._state.update(t=time.time(), ok=False, why="Ollama error; using built-in answer")
            note = "\n\n(The local AI model stopped, so here is the built-in answer.)\n\n" if out else ""
            out = (out + note + reply) if out else reply
            yield ev(t=(note + reply))
            used = "builtin"
        db.save(u, "chat", dict(q=q, a=out, intent=intent, conf=conf, engine=used), q[:200])
        yield ev(done=True, engine=used, intent=intent)

    return Response(gen(), mimetype="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ history
@app.route("/history")
@login_required
def history():
    kind = request.args.get("kind")
    if kind and kind not in ("soil", "weather", "irrigation", "crop", "fertilizer", "scan"):
        kind = None
    rows = [r for r in db.recent(uid(), kind, 300) if r["kind"] != "chat"]
    return render_template("history.html", rows=rows, kind=kind)


# ------------------------------------------------------------------ admin
@app.route("/admin", methods=["GET", "POST"])
@login_required
@admin_required
def admin():
    if request.method == "POST":
        try:
            target = int(request.form["uid"])
        except (KeyError, ValueError):
            abort(400)
        act = request.form.get("action")
        with db.Sess() as s:
            t = s.get(db.User, target)
            if t and t.id != session["user"]["id"]:
                if act == "toggle":
                    t.active = not t.active
                elif act == "role" and request.form.get("role") in ("farmer", "agronomist", "admin"):
                    t.role = request.form["role"]
                elif act == "delete":
                    s.query(db.Record).filter_by(user_id=t.id).delete()
                    s.query(db.Farm).filter_by(user_id=t.id).delete()
                    s.delete(t)
                s.commit()
            else:
                flash("Cannot modify yourself or a missing user.")
    p = pd.read_sql(text("select id,name,email,role,active,created from users order by id"), db.eng)
    return render_template("admin.html", cols=list(p.columns), rows=p.values.tolist(), report=REPORT)


@app.route("/database")
@login_required
@admin_required
def database():
    t = request.args.get("table", "users")
    t = t if t in ("users", "records", "farms") else "users"
    cols = {"users": "id,name,email,role,active,created", "records": "id,user_id,kind,created,summary", "farms": "id,user_id,region,soil,area_ha,crop,sow_date,method"}[t]
    if request.args.get("export"):
        d = pd.read_sql(text(f"select {cols} from {t} order by id"), db.eng)
        return Response(d.to_csv(index=False), mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename={t}.csv"})
    d = pd.read_sql(text(f"select {cols} from {t} order by id desc limit 200"), db.eng)
    total = pd.read_sql(text(f"select count(*) n from {t}"), db.eng)["n"][0]
    return render_template("db.html", t=t, cols=list(d.columns), rows=d.values.tolist(), total=int(total))


@app.template_filter("pct")
def pct(v):
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


if __name__ == "__main__":
    app.run(debug=False, port=int(os.getenv("PORT", 8000)))
