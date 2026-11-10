"""Train every local model and write models/*.joblib + models/report.json.

    python train.py                       # default: datasets generated from data/*.json
    python train.py --crop-csv crops.csv  # real crop data: N,P,K,ph,temp,humidity,rain,soil,label
    python train.py --weather-csv x.csv --region east_humid   # daily date,tmax,tmin,rh,rain,pres
    python train.py --image-dir leaves/   # folders named like keys in data/diseases.json
"""
import argparse
import json
import os
import time
import pandas as pd
from engines import MODELS, crop, weather, vision, chatbot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-csv")
    ap.add_argument("--weather-csv")
    ap.add_argument("--region", default="east_humid")
    ap.add_argument("--image-dir")
    ap.add_argument("--only", choices=["crop", "weather", "vision", "chatbot"])
    a = ap.parse_args()
    os.makedirs(MODELS, exist_ok=True)
    rep = {}
    t = time.time()
    if a.only in (None, "crop"):
        if a.crop_csv:
            df = pd.read_csv(a.crop_csv)
            m = crop.train(df.drop(columns=["label"]), df["label"].values)
        else:
            m = crop.train()
        rep["crop"] = m["metrics"]
        print("crop", m["metrics"], f"{time.time() - t:.0f}s", flush=True)
    if a.only in (None, "weather"):
        t = time.time()
        csv = None
        if a.weather_csv:
            d = pd.read_csv(a.weather_csv, parse_dates=["date"])
            csv = {a.region: d}
        m = weather.train(region_csv=csv)
        rep["weather"] = m["metrics"]
        print("weather", m["metrics"], f"{time.time() - t:.0f}s", flush=True)
    if a.only in (None, "vision"):
        t = time.time()
        if a.image_dir:
            X, y = vision.image_dir_dataset(a.image_dir)
            m = vision.train(X, y, source=f"image folder {a.image_dir}")
        else:
            m = vision.train()
        met = dict(m["metrics"])
        met.pop("confusion", None)
        rep["vision"] = met
        print("vision", met, f"{time.time() - t:.0f}s", flush=True)
    if a.only in (None, "chatbot"):
        t = time.time()
        m = chatbot.train()
        rep["chatbot"] = m["metrics"]
        print("chatbot", m["metrics"], f"{time.time() - t:.0f}s", flush=True)
    path = os.path.join(MODELS, "report.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(rep)
    json.dump(old, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
