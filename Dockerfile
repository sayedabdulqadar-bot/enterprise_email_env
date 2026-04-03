FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir pydantic openai fastapi uvicorn requests python-dotenv pyyaml

# Copy application files
COPY server/ /app/server/
COPY data/ /app/data/

# Ensure app.py exists
RUN ls -la /app/server/ || echo "Server directory contents checked"

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV OPENENV_HOST=0.0.0.0
ENV OPENENV_PORT=7860

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health', timeout=5)" || exit 1

# Start the app
CMD ["python", "-u", "server/app.py"]
