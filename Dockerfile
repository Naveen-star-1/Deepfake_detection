FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for handling Audio files (Librosa needs these)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them securely
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the Docker container
COPY . .

# Hugging Face Spaces route to port 7860 by default for Docker environments
EXPOSE 7860

# Command to run the Deepfake Streamlit App
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
