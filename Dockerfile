# Base Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Set the working directory in the container
WORKDIR $APP_HOME

# Create a non-root user and group
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Install system dependencies
# build-essential is useful for packages that compile C extensions
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
# Using a virtual environment within Docker is generally not necessary as the container itself is isolated.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model
# This ensures it's part of the image. Ensure this matches settings.spacy_model_name.
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code into the container
# This should be done after installing dependencies to leverage caching better.
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser $APP_HOME

# Switch to the non-root user
USER appuser

# Expose the port the app runs on (as defined in app/main.py and CMD)
EXPOSE 8000

# Define the command to run the application
# This will run the FastAPI server using Uvicorn.
# Ensure app.main:app points to your FastAPI instance.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 