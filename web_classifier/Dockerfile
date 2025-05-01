# Dockerfile
FROM python:3.9-slim

# Set up a non-root user with UID 1000
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g 1000 -ms /bin/bash appuser

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates templates/

# Create necessary directories and set permissions
RUN mkdir -p data/annotations && \
    chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

CMD ["python", "app.py"]