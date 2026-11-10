# AgriXAI: local AI smart-agriculture platform

Flask web app where **every AI feature runs on your own server**: scikit-learn models, OpenCV computer vision, rule engines and a TF-IDF NLP chatbot. No Claude/OpenAI/Gemini or any other AI API, no API keys.

| Feature | How it works |
|---|---|
| Weather prediction | Gradient-boosting models (Tmax, Tmin, humidity, pressure, rain probability, rain amount) on lag + seasonal features, rolled forward 7 days. Start from regional climate normals or your own observations. |
| Soil analysis | Soil Health Card style N/P/K/OC ratings, pH/EC classes, moisture vs field capacity, weighted health index, prioritised actions. |
| Smart irrigation | FAO-56 water balance: Hargreaves ETo from forecast, stage-based Kc, root-zone depletion, effective rain, method efficiency; gives a 7-day schedule in mm and litres. |
| Crop recommendation | Random forest + gradient boosting ensemble blended with agronomic suitability rules; every crop is ranked with the factors that help or limit it. |
| Crop image scanner | OpenCV leaf segmentation + ~50 colour/lesion/hole/edge/mottling features → RF + MLP ensemble. Detects 8 problems (blight, leaf spot, powdery mildew, rust, N and K deficiency, pest damage, mosaic virus) or healthy; severity from affected area; treatment and prevention advice. |
| Fertilizer | Crop dose adjusted by soil-test rating, converted to DAP/urea/MOP quantities, split schedule by growth stage, lime/gypsum/sulphur/zinc/organic corrections. |
| Chatbot | **Your local Ollama model (llama3.2:3b)** answers in plain language, grounded in your farm data, rule-engine numbers and retrieved knowledge; if Ollama is off it falls back to the built-in NLP bot: TF-IDF + logistic-regression intents, entity extraction, TF-IDF retrieval over a 29-passage knowledge base; answers are composed from *your* latest soil report, forecast, irrigation, scan and history. |
| Dashboard | Charts and cross-feature insights (e.g. fungal disease found + heavy rain forecast). |

## Run locally
```bash
pip install -r requirements.txt
python train.py          # ~3 min: trains all models into models/
python app.py            # http://localhost:8000  (SQLite agri.db)
pytest tests             # end-to-end tests of every feature
```
If `ADMIN_PASSWORD` is not set, a random admin password is printed to the console on first start.

## Chatbot with Ollama (local LLM)
```bash
ollama pull llama3.2:3b     # once
ollama serve                # usually already running in the background
python app.py               # the Chat page shows 🟢 Local AI when Ollama is detected
```
How it works: the question goes through the built-in NLP + rule engines first (soil ratings, crop ranking, fertiliser doses, irrigation plan computed from *your* data). Those facts, the retrieved knowledge passages and recent chat turns are sent to Ollama as grounding, and the answer streams back token by token. The model is told not to invent doses, dates or chemicals. Settings (env): `OLLAMA_URL` (default `http://127.0.0.1:11434`), `OLLAMA_MODEL` (default `llama3.2:3b`), `OLLAMA_ENABLED=0` to turn it off, `OLLAMA_TIMEOUT`. With Docker on your PC use `OLLAMA_URL=http://host.docker.internal:11434`.
A 3B model can still make mistakes, so pesticide advice always tells the farmer to check the label and ask a local agriculture officer.
**Render:** Ollama cannot run on Render's free tier, so the deployed site automatically uses the built-in NLP bot. To use the LLM online, host Ollama on a machine you control (a VM with enough RAM) and set `OLLAMA_URL` to it.

## Deploy on Render
1. Push this folder to GitHub. 2. Render → New → Blueprint → select the repo (`render.yaml` creates the web service and Postgres). 3. Set `ADMIN_EMAIL` and `ADMIN_PASSWORD` when prompted (`SECRET_KEY` is generated). The Docker build trains the models; memory use is ≈270 MB (fits the free tier). Free Render Postgres databases expire after 30 days, so upgrade or export data for anything long-term.

## Data and accuracy: read this
No internet datasets are bundled, so the defaults are built from knowledge in `data/*.json` (crop requirement ranges, soil properties, regional climate normals, disease knowledge, fertiliser products, chatbot intents/passages):
* **Weather / crop / vision models are trained on generated data** (a stochastic weather generator from regional normals; samples drawn from crop requirement ranges; procedurally rendered leaves). The reported accuracies measure how well they learned *that* data, not real-world accuracy. Retrain on real data for field use:
  * `python train.py --crop-csv crops.csv` (columns `N,P,K,ph,temp,humidity,rain,soil,label`; e.g. the public Kaggle crop-recommendation data)
  * `python train.py --weather-csv obs.csv --region east_humid` (daily `date,tmax,tmin,rh,rain,pres` from IMD/NASA POWER/Open-Meteo archives)
  * `python train.py --image-dir leaves/` (one folder per class named as in `data/diseases.json`, e.g. PlantVillage re-labelled)
* Without your own observations the weather forecast is essentially seasonal climatology with day-to-day dynamics; enter today's readings on the Weather page to localise it. It is not a substitute for official weather warnings.
* TensorFlow/PyTorch are deliberately not used: they exceed Render's free memory/build limits and scikit-learn models are sufficient here. The vision feature extractor can be swapped for a CNN later.

## Security notes
CSRF on all POSTs, bcrypt passwords, login throttling, secure cookies on Render, upload size/type validation, per-user data isolation, no raw-SQL console (the old regex-guarded one was bypassable). Never commit `.env`; if a real `.env` was shared anywhere, change those passwords.
