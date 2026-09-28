FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py models.py ./


# Download models and data at BUILD TIME, not at container startup -
# Cloud Run can spin up new instances frequently (scale up/down), so
# downloading at runtime would slow down every cold start and depend
# on Hugging Face Hub being available in production
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('model_files/all-MiniLM-L6-v2')"

EXPOSE 8000

# $PORT is provided dynamically by the hosting platform (Cloud Run/Render), not fixed at 8000
CMD uvicorn main:app --host 0.0.0.0 --port $PORT