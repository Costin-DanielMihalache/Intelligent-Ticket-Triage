FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py models.py ./

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mihalachecostindaniel/ticket-classifier-distilbert', local+dir='model_files/ticket-classifier-distilbert'); snapshot_download(repo+id='mihalachecostindaniel/ticket-triage-data', repo_type='dataset', local_dir='data')"

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT