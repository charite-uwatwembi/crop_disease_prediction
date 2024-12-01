
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Create necessary directories
RUN mkdir -p data/new_data/retrain static

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Upgrade pip and install dependencies in a single layer to reduce build time
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt --index-url https://pypi.org/simple

# Copy only the application code and models (separate from requirements to leverage caching)
COPY src /app/src
COPY models /app/models

# Set Python path
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Default command to run the app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
