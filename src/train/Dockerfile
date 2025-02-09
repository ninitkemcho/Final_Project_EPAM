# Dockerfile for training
FROM python:3.8

# Changing the path to the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# Ensure necessary NLTK resources are available
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('punkt_tab'); \
    nltk.download('wordnet'); \
    nltk.download('stopwords'); \
    nltk.download('omw-1.4')"

# Copy data folder to Docker
COPY data /app/data/

# Copy the training script
COPY src/train /app/src/train/

# Train the model
RUN python3 /app/src/train/train.py
