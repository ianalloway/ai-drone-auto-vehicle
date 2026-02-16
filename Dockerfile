FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for drone libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir dronekit pymavlink numpy

# Copy application code
COPY . .

# Expose MAVLink port
EXPOSE 14550

# Default command - can be overridden
CMD ["python", "-m", "src.main"]
