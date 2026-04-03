ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Set Python path
ENV PYTHONPATH="/app"
ENV PORT=8000
ENV MAX_CONCURRENT_ENVS=8

# Start server
CMD ["sh", "-c", "cd /app && uvicorn env.server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
