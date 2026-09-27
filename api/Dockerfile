FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py models.py ./

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mihalachecostindaniel/ticket-classifier-distilbert', local_dir='model_files/ticket-classifier-distilbert'); snapshot_download(repo_id='mihalachecostindaniel/ticket-triage-data', repo_type='dataset', local_dir='data'); snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', local_dir='model_files/all-MiniLM-L6-v2')"

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT