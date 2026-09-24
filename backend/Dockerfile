FROM python:3.11-slim

WORKDIR /app

# Install deps in a separate layer (cache-friendly)
COPY requirements.txt requirements-aws.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-aws.txt

# Copy the API source — rename to underscores (Python cannot import hyphenated modules)
COPY radar-api-main.py radar_api_main.py

# Copy the activity sub-package (XGBoost model + detector)
COPY activity/ ./activity/

EXPOSE 80

# Run uvicorn directly on port 80 (nginx is not needed inside Docker)
CMD ["uvicorn", "radar_api_main:app", "--host", "0.0.0.0", "--port", "80"]
