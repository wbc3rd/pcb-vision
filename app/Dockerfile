# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment (optional but recommended)
# RUN python3 -m venv /env

# Install the required Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port the app will run on
EXPOSE 8501

# Set the environment variable for the container
ENV PYTHONUNBUFFERED=1

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "app.py"]
