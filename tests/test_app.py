"""End-to-end tests: every feature through the real Flask app with real models (no mocks)."""
import io
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/t.db"
os.environ["ADMIN_EMAIL"] = "admin@test.local"
os.environ["ADMIN_PASSWORD"] = "AdminPass#123"
os.environ.pop("RENDER", None)

import cv2
import numpy as np
import pytest
import app as A
from engines import vision


@pytest.fixture(scope="module")
def client():
    A.app.config["TESTING"] = True
    c = A.app.test_client()
    tok = lambda: re.search(r'name="csrf" value="(\w+)"', c.get("/login").get_data(as_text=True)).group(1)
    r = c.post("/register", data=dict(csrf=tok(), name="Test Farmer", email="f@test.com", password="password1", confirm="password1"))
    assert r.status_code == 302
    r = c.post("/login", data=dict(csrf=tok(), email="f@test.com", password="password1"))
    assert r.status_code == 302
    c.tok = lambda: re.search(r'name="csrf" value="(\w+)"', c.get("/farm").get_data(as_text=True)).group(1)
    return c


def post(c, url, **d):
    return c.post(url, data=dict(csrf=c.tok(), **d), follow_redirects=True)


def test_csrf_and_auth():
    c = A.app.test_client()
    assert c.get("/dashboard").status_code == 302
    assert c.post("/login", data=dict(email="a", password="b")).status_code == 400   # missing CSRF


def test_all_pages_render(client):
    for u in ["/dashboard", "/farm", "/weather", "/soil", "/irrigation", "/crops", "/fertilizer", "/scanner", "/chat", "/history"]:
        r = client.get(u)
        assert r.status_code == 200, u


def test_farm_profile(client):
    r = post(client, "/farm", region="east_humid", soil="alluvial", area_ha="2.5", crop="rice", sow_date="2026-07-01", method="drip")
    assert "Farm profile saved" in r.get_data(as_text=True)


def test_weather(client):
    r = post(client, "/weather", region="east_humid", tmax="31", tmin="24", rh="82", pres="", rain="4")
    h = r.get_data(as_text=True)
    assert "Farming outlook" in h and h.count("<tr><td>20") >= 7
    from db import latest
    d = latest(2, "weather")["data"]
    assert len(d["days"]) == 7
    for day in d["days"]:
        assert 0 <= day["rain_prob"] <= 100 and 0 <= day["rh"] <= 100 and day["tmin"] < day["tmax"]


def test_soil_poor_and_good(client):
    r = post(client, "/soil", soil="red", ph="4.8", moisture="10", N="120", P="5", K="60", oc="0.3", ec="")
    h = r.get_data(as_text=True)
    assert "Poor" in h and "strongly acidic" in h.lower() or "Strongly acidic" in h
    r = post(client, "/soil", soil="loamy", ph="6.8", moisture="25", N="450", P="30", K="300", oc="0.9", ec="0.5")
    assert "Good" in r.get_data(as_text=True)


def test_validation_rejects_bad_input(client):
    assert post(client, "/soil", soil="red", ph="99", moisture="10", N="1", P="1", K="1", oc="1", ec="").status_code == 400
    assert post(client, "/soil", soil="evil", ph="6", moisture="10", N="1", P="1", K="1", oc="1", ec="").status_code == 400
    assert post(client, "/soil", soil="red", ph="abc", moisture="10", N="1", P="1", K="1", oc="1", ec="").status_code == 400


def test_irrigation_logic(client):
    r = post(client, "/irrigation", crop="wheat", soil="sandy", region="north_plains", moisture="6", sow_date="2026-10-20", method="flood", area_ha="1")
    assert "Irrigate now" in r.get_data(as_text=True)
    r = post(client, "/irrigation", crop="wheat", soil="loamy", region="north_plains", moisture="28", sow_date="2026-10-20", method="drip", area_ha="1")
    assert r.status_code == 200
    r = post(client, "/irrigation", crop="wheat", soil="clay", region="north_plains", moisture="55", sow_date="2026-10-20", method="drip", area_ha="1")
    assert "Do not irrigate" in r.get_data(as_text=True)


def test_irrigation_math():
    from engines import irrigation as I
    # Hargreaves ETo for a hot dry day at 28N in June is a few mm/day
    e = I.eto(40, 27, 28.6, 170)
    assert 6 < e < 11
    assert I.kc_at(I.CROPS["wheat"], 0)[1] == "initial" and I.kc_at(I.CROPS["wheat"], 60)[1] == "mid-season"
    assert I.eff_rain(1) == 0 and 15 < I.eff_rain(30) < 30


def test_crops(client):
    r = post(client, "/crops", N="90", P="45", K="45", ph="6.4", temp="26", humidity="85", rain="1500", soil="clay")
    h = r.get_data(as_text=True)
    assert "Best match" in h
    from db import latest
    top = latest(2, "crop")["data"]["top"]
    assert top[0]["score"] >= top[1]["score"]
    r = post(client, "/crops", N="100", P="40", K="40", ph="7.0", temp="14", humidity="50", rain="450", soil="loamy")
    from db import latest as L
    assert L(2, "crop")["data"]["top"][0]["key"] in ("wheat", "mustard", "chickpea", "potato")


def test_fertilizer(client):
    r = post(client, "/fertilizer", crop="wheat", soil="red", ph="5.0", N="200", P="8", K="90", oc="0.4", ec="", zn="", area_ha="2", sow_date="2026-11-10")
    h = r.get_data(as_text=True)
    assert "Lime" in h and "Urea" in h and "DAP" in h
    from db import latest
    d = latest(2, "fertilizer")["data"]
    assert d["ratings"] == {"N": "Low", "P": "Low", "K": "Low"} and d["dose"]["N"] == 150
    # sodic soil -> gypsum
    r = post(client, "/fertilizer", crop="wheat", soil="clay", ph="9.0", N="300", P="15", K="150", oc="0.6", ec="", zn="", area_ha="1", sow_date="")
    assert "Gypsum" in r.get_data(as_text=True)


def _leaf(cls, seed):
    img = vision.synthetic_leaf(cls, np.random.default_rng(seed))
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.mark.parametrize("cls", ["healthy", "rust", "powdery_mildew", "nitrogen_deficiency", "pest_damage", "leaf_spot"])
def test_scanner_detects(client, cls):
    hits = 0
    for seed in (101, 102, 103, 104, 105):
        res = vision.scan(A.M_VISION, _leaf(cls, seed))
        assert res["status"] == "ok"
        hits += res["label"] == cls
    assert hits >= 3, f"{cls}: {hits}/5"


def test_scanner_upload_and_history(client):
    r = client.post("/scanner", data=dict(csrf=client.tok(), crop="rice", image=(io.BytesIO(_leaf("rust", 7)), "leaf.jpg")), content_type="multipart/form-data")
    h = r.get_data(as_text=True)
    assert r.status_code == 200 and "Treatment" in h and "/scan-image/" in h
    rid = int(re.search(r"/scan-image/(\d+)", h).group(1))
    assert client.get(f"/scan-image/{rid}").mimetype == "image/jpeg"


def test_scanner_rejects_non_leaf_and_garbage(client):
    blank = cv2.imencode(".png", np.full((200, 200, 3), 240, np.uint8))[1].tobytes()
    r = client.post("/scanner", data=dict(csrf=client.tok(), image=(io.BytesIO(blank), "b.png")), content_type="multipart/form-data")
    assert "No leaf" in r.get_data(as_text=True)
    r = client.post("/scanner", data=dict(csrf=client.tok(), image=(io.BytesIO(b"not an image"), "x.jpg")), content_type="multipart/form-data")
    assert r.status_code == 400


def chat(c, msg):
    r = c.post("/api/chat", json=dict(message=msg), headers={"X-CSRF": c.tok()})
    assert r.status_code == 200
    return r.get_json()


def test_chatbot_personalised(client):
    j = chat(client, "How is my soil?")
    assert j["intent"] == "my_soil" and "health index" in j["reply"]
    j = chat(client, "will it rain this week")
    assert j["intent"] == "my_weather" and "mm" in j["reply"]
    j = chat(client, "should i irrigate today")
    assert j["intent"] == "irrigation"
    j = chat(client, "what fertilizer should i use for wheat")
    assert j["intent"] == "fertilizer" and "kg/ha" in j["reply"]
    j = chat(client, "explain my last scan")
    assert j["intent"] == "scan_result" and "confidence" in j["reply"]
    j = chat(client, "which crop should i grow")
    assert j["intent"] == "crop_recommend"
    j = chat(client, "how to control powdery mildew")
    assert "sulphur" in j["reply"].lower()
    j = chat(client, "how to make vermicompost")
    assert "earthworm" in j["reply"].lower() or "vermicompost" in j["reply"].lower()
    j = chat(client, "how to grow potato")
    assert "potato" in j["reply"].lower()
    assert len(chat(client, "show my history")["reply"]) > 10


def test_chat_input_limits(client):
    assert client.post("/api/chat", json=dict(message="x" * 600), headers={"X-CSRF": client.tok()}).status_code == 400
    assert client.post("/api/chat", json=dict(message="hi")).status_code == 400   # no CSRF


def test_dashboard_has_everything(client):
    h = client.get("/dashboard").get_data(as_text=True)
    for s in ("Soil health", "Irrigation", "Best crop now", "Last crop scan", "Farming insights", "Local models"):
        assert s in h


def test_history_and_isolation(client):
    assert "scan" in client.get("/history").get_data(as_text=True)
    other = A.app.test_client()
    tok = lambda: re.search(r'name="csrf" value="(\w+)"', other.get("/login").get_data(as_text=True)).group(1)
    other.post("/register", data=dict(csrf=tok(), name="Other", email="o@test.com", password="password1", confirm="password1"))
    other.post("/login", data=dict(csrf=tok(), email="o@test.com", password="password1"))
    assert other.get("/scan-image/1").status_code == 404           # cannot read someone else's image
    assert other.get("/admin").status_code == 403 and other.get("/database").status_code == 403


def test_admin(client):
    a = A.app.test_client()
    tok = lambda: re.search(r'name="csrf" value="(\w+)"', a.get("/login").get_data(as_text=True)).group(1)
    assert a.post("/login", data=dict(csrf=tok(), email="admin@test.local", password="AdminPass#123")).status_code == 302
    assert a.get("/admin").status_code == 200
    h = a.get("/database?table=users").get_data(as_text=True)
    assert "f@test.com" in h and "$2b$" not in h
    csv = a.get("/database?table=records&export=1")
    assert csv.mimetype == "text/csv" and b"summary" in csv.data


# ---------------------------------------------------------------- Ollama integration (against a local stub server speaking Ollama's HTTP API)
import json as _json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

SEEN = {}


class _Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = _json.dumps({"models": [{"name": "llama3.2:3b"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        req = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN["req"] = req
        self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
        for tok in ["Grow ", "wheat. ", "Add lime."]:
            self.wfile.write((_json.dumps({"message": {"role": "assistant", "content": tok}, "done": False}) + "\n").encode())
        self.wfile.write((_json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}) + "\n").encode())


def _stream(c, q):
    r = c.post("/api/chat/stream", json=dict(message=q), headers={"X-CSRF": c.tok()})
    assert r.status_code == 200
    ev = [_json.loads(l) for l in r.get_data(as_text=True).splitlines() if l.strip()]
    return "".join(e.get("t", "") for e in ev), ev[-1]


def test_chat_stream_fallback_when_ollama_off(client):
    from engines import llm
    llm.URL = "http://127.0.0.1:9"; llm._state.update(t=0, ok=False)
    text, last = _stream(client, "which crop should i grow")
    assert last["done"] and last["engine"] == "builtin" and "score" in text
    assert client.get("/api/chat/status").get_json()["mode"] == "builtin"


def test_chat_stream_with_ollama_grounded(client):
    from engines import llm
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    llm.URL = f"http://127.0.0.1:{srv.server_port}"; llm._state.update(t=0, ok=False)
    assert client.get("/api/chat/status").get_json()["mode"] == "ollama"
    text, last = _stream(client, "what is needed for better cultivation of my best crop?")
    assert text == "Grow wheat. Add lime." and last["engine"] == "ollama"
    sysmsg = SEEN["req"]["messages"][0]["content"]
    # the prompt must carry the farmer's real data + engine-computed numbers + retrieved knowledge
    for needle in ("FARM DATA", "Soil report", "Crop ranking", "Fertiliser plan", "VERIFIED ANSWER", "KNOWLEDGE"):
        assert needle in sysmsg, needle
    assert SEEN["req"]["model"] == "llama3.2:3b" and SEEN["req"]["messages"][-1]["role"] == "user"
    srv.shutdown()
    llm.URL = "http://127.0.0.1:9"; llm._state.update(t=0, ok=False)
