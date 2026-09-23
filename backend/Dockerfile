FROM python:3.11-slim

WORKDIR /app

# Install dependencies first — separate layer so code changes don't bust this cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API source (renamed locally from radar-api-main.backup.py → radar-api.py)
COPY radar-api.py .

# Expose FastAPI on port 80 (matches existing EC2 / NGINX setup)
EXPOSE 80

# Run with uvicorn (production ASGI server)
CMD ["uvicorn", "radar-api:app", "--host", "0.0.0.0", "--port", "80"]
