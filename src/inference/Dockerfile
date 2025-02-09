FROM python:3.8

WORKDIR /app

# Copy model to the correct location
COPY outputs/models/model.pkl /app/outputs/models/model.pkl

# Copy processed data
COPY data/processed/ /app/data/processed/

# Copy inference script
COPY src/inference/ /app/src/inference/

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Ensure outputs directories exist
RUN mkdir -p /app/outputs/predictions /app/outputs/figures

# Set default command to run inference when the container starts
RUN python3 /app/src/inference/run_inference.py
