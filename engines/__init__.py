"""Local AI engines for AgriXAI. Nothing here calls an external AI service."""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")


def load_json(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def trap(x, a, b, c, d):
    """Trapezoidal suitability membership in [0, 1]: 0 outside [a, d], 1 within [b, c]."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    return (x - a) / (b - a) if x < b else (d - x) / (d - c)
