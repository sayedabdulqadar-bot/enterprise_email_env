FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir fastapi uvicorn pydantic requests python-dotenv pyyaml

# Copy project files
COPY env/ /app/env/
COPY data/ /app/data/

# Environment setup
ENV PYTHONUNBUFFERED=1

# HF uses dynamic PORT (VERY IMPORTANT)
ENV PORT=7860

EXPOSE 7860

# ❌ DO NOT USE HEALTHCHECK (causes restart loop on HF)

# ✅ Start FastAPI correctly using uvicorn
CMD ["sh", "-c", "uvicorn env.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]