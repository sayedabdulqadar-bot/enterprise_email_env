FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir fastapi uvicorn pydantic requests python-dotenv pyyaml

# ✅ CORRECT PATH (your structure)
COPY env/ /app/env/
COPY data/ /app/data/

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

# ❌ NO HEALTHCHECK

# ✅ CORRECT START
CMD ["sh", "-c", "uvicorn env.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]