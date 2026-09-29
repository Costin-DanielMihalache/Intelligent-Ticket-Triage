import torch
import numpy as np
import faiss
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from dotenv import load_dotenv
import os
from google import genai
import time
import logging
import json
from datetime import datetime
from huggingface_hub import hf_hub_download
from pathlib import Path
import random
from google.genai import types
import sys
import hashlib

torch.set_grad_enabled(False) # inference only - no training here, saves memory/time
torch.set_num_threads(1) # avoid over-allocating threads on resource-limited Cloud run instances

BASE_DIR= Path(__file__).resolve().parent
MODEL_PATH=str(BASE_DIR/"model_files"/"ticket-classifier-distilbert")

tokenizer=DistilBertTokenizer.from_pretrained(MODEL_PATH)
model_bert=DistilBertForSequenceClassification.from_pretrained(MODEL_PATH,torch_dtype=torch.float16)
# float16 instead of float32: halves the model's memory footprint with minimal
# accuracy impact - important on memory-limited cloud instances
model_bert.eval()

MODEL_DIR=BASE_DIR/"model_files"/"all-MiniLM-L6-v2"
embedder= SentenceTransformer(str(MODEL_DIR))

DATA_DIR=BASE_DIR/"data"
df_resolved= pd.read_csv(DATA_DIR/"df_resolved.csv")
ticket_embeddings=np.load(DATA_DIR/"ticket_embeddings.npy")

dimension=ticket_embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(ticket_embeddings.astype('float32'))

label_classes=sorted(df_resolved['queue'].unique())
label_encoder=LabelEncoder()
label_encoder.fit(df_resolved['queue'])

def predict_category_with_confidence(text):
    inputs=tokenizer(text,truncation=True,padding=True,max_length=256,return_tensors='pt')
    inputs={k: v.to(model_bert.device) for k,v in inputs.items()}

    with torch.no_grad():
        outputs=model_bert(**inputs)
    probs=F.softmax(outputs.logits,dim=1)
    confidence,predicted_label=torch.max(probs,1)

    predicted_category=label_encoder.inverse_transform([predicted_label.item()])[0]
    return predicted_category, confidence.item()

def find_similar_tickets_hybrid(query_text,k=5, confidence_threshold=0.5):
    predicted_category, confidence= predict_category_with_confidence(query_text)
    query_embedding= embedder.encode([query_text]).astype('float32')

    if confidence >= confidence_threshold:
        # Model is confident enough about the category -> search only within it
        # (more relevant, but may miss good results from other categories)
        mask=df_resolved['queue']==predicted_category
        filtered_df=df_resolved[mask].reset_index(drop=True)
        filtered_embeddings=ticket_embeddings[mask.values].astype('float32')

        temp_index=faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)

        distances, indices= temp_index.search(query_embedding, min(k,len(filtered_df)))
        source_df=filtered_df
        mode="filtered"
    else:
        # Model is uncertain -> search across the whole dataset, to avoid
        # missing a relevant result just because classification was unsure
        distances,indices=index.search(query_embedding,k)
        source_df=df_resolved
        mode="unfiltered"

    results=[]
    for i,idx in enumerate(indices[0]):
        results.append({
            'subject':source_df.iloc[idx]['subject'],
            'queue':source_df.iloc[idx]['queue'],
            'answer':source_df.iloc[idx]['answer'],
            'distance':distances[0][i]
        })
    return predicted_category,confidence,mode,results

load_dotenv()
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")
client_gemini=genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=10_000))

def generate_response(ticket_text,similar_results):
    context=""
    for i, r in enumerate(similar_results):
        context+=f"\nExample {i+1}:\nSimilar ticket: {r['subject']}\nSolution used: {r['answer']}\n"
    prompt=f"""You are a customer support assistant. You received a new ticket from a customer:
    
"{ticket_text}"

Here are some similar cases resolved previously, which can help you formulate a response:
{context}

Write a professional and empathetic response to the new ticket, inspired by the examples above, but adapted specifically to this case."""

    response= client_gemini.models.generate_content(
        model='gemini-3.5-flash-lite',
        contents=prompt
    )
    return response.text

RETRYABLE_CODES={429,500,502,503,504}

def _is_retryable(exc):
    # google-genai API errors expose the HTTP status as `.code`; errors without
    # one (timeouts, connection resets) are treated as transient. Client errors
    # like 400/401/403 (bad request, bad key) will never succeed on retry.
    code=getattr(exc,"code",None)
    return code is None or code in RETRYABLE_CODES

def generate_response_with_retry(ticket_text,similar_results,max_retries=3,base_delay=1.0,max_delay=8.0):
    for attempt in range(max_retries):
        try:
            return generate_response(ticket_text,similar_results)
        except Exception as e:
            retryable=_is_retryable(e)
            logging.getLogger('ticket_triage').warning(
                "Gemini call failed (attempt %d/%d, retryable=%s): %s",
                attempt+1,max_retries,retryable,e)
            if not retryable or attempt==max_retries-1:
                return None
            # Exponential backoff with full jitter: the random spread stops many
            # instances from retrying in lockstep against a rate-limited API
            time.sleep(random.uniform(0,min(max_delay,base_delay*2**attempt)))
    return None

def process_ticket(ticket_text):
    category, confidence, mode, results=find_similar_tickets_hybrid(ticket_text)
    response=generate_response_with_retry(ticket_text,results)
    return {
        'category': category,
        'confidence': confidence,
        'mode': mode,
        'generated_response': response
    }

def route_ticket(ticket_text,confidence_threshold=0.5,distance_threshold=0.85):
    # Thresholds below were chosen empirically, based on observing confidence/
    # distance scores on test examples (not a rigorous optimization) - a clear
    # candidate for future calibration
    category,confidence,mode, results=find_similar_tickets_hybrid(ticket_text)
    min_distance=min(r['distance'] for r in results) if results else float('inf')

    if confidence >= confidence_threshold and min_distance <= distance_threshold:
        # Only auto-resolve if the model is CONFIDENT about the category AND
        # found a sufficiently similar past case - either condition failing
        # means an automatic response would be risky
        decision="AUTO_RESOLVE"
        response=generate_response_with_retry(ticket_text,results)
        if response is None:
            # Gemini unavailable after retries: a ticket marked AUTO_RESOLVE with
            # no answer would silently drop the customer, so hand it to a human
            decision = "ESCALATE_TO_HUMAN"
    else:
        decision="ESCALATE_TO_HUMAN"
        response=None

    return {
        'ticket': ticket_text,
        'category': category,
        'confidence': float(confidence),
        'retrieval_mode': mode,
        'min_distance': float(min_distance),
        'decision': decision,
        'generated_response': response
    }

class JsonFormatter(logging.Formatter):
    #One JSON object per line on stdout: Cloud Logging turns  it into a structured
    #entry and reads the "severity" key, so filtering by WARNING/ERROR works
    def format(self,record):
        payload={"severity":record.levelname,"message":record.getMessage()}
        payload.update(getattr(record,"fields",{}))
        return json.dumps(payload)

logger= logging.getLogger('ticket_triage')
logger.setLevel(logging.INFO)
logger.propagate=False

if not logger.handlers:
    handler=logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


def route_ticket_with_logging(ticket_text,confidence_threshold=0.5, distance_threshold=0.85):
    result=route_ticket(ticket_text,confidence_threshold,distance_threshold)
    # Never log the ticket body: real tickets can contain personal data.
    # A short hash still lets us correlate repeats without storing the content.
    log_entry={
        'ticket_hash':hashlib.sha256(ticket_text.encode()).hexdigest()[:12],
        'ticket_length':len(ticket_text),
        'category':result['category'],
        'confidence':round(float(result['confidence']),3),
        'retrieval_mode':result['retrieval_mode'],
        'min_distance':round(float(result['min_distance']),3),
        'decision':result['decision']
    }

    logger.info("ticket_routed",extra={"fields":log_entry})
    return result