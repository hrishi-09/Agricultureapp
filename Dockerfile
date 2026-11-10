FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Train every local model at build time (weather, crop, vision, chatbot): no downloads, no external AI
RUN python train.py
RUN useradd -m app && chown -R app /app
USER app
CMD ["sh","-c","gunicorn app:app -b 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120"]
