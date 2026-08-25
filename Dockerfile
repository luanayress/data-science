# Minimal, production-safe Dockerfile for API serving
# Use official Python image with deterministic version
FROM python:3.10-slim

# Set environment variables for artifact/config paths (override at runtime)
ENV MODEL_DIR=/app/models \
    CONFIG_DIR=/app/configs \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs

WORKDIR /app

# Install Python dependencies (deterministic)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose API port
EXPOSE 8000

# Default runtime is FastAPI inference server.
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
