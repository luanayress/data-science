# Minimal, production-safe Dockerfile for ML batch training/inference
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

# Entrypoint for batch training (override CMD for inference)
ENTRYPOINT ["python", "src/pipelines/training_pipeline.py"]
# For inference, override with: python -m app.app or similar

# Expose port for API if needed (optional)
# EXPOSE 8080
