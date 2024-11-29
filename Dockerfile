# Use the official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Create necessary directories
RUN mkdir -p data/new_data/retrain
RUN mkdir -p static

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY src /app/src
COPY models /app/models

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Default command to run the app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
