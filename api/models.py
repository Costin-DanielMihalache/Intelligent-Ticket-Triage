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


MODEL_PATH="mihalachecostindaniel/ticket-classifier-distilbert"

tokenizer=DistilBertTokenizer.from_pretrained(MODEL_PATH)
model_bert=DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model_bert.eval()

embedder= SentenceTransformer('all-MiniLM-L6-v2')

df_path=hf_hub_download(repo_id="mihalachecostindaniel/ticket-triage-data",filename="df_resolved.csv",repo_type="dataset")
embeddings_path=hf_hub_download(repo_id="mihalachecostindaniel/ticket-triage-data",filename="ticket_embeddings.npy",repo_type="dataset")

df_resolved= pd.read_csv(df_path)
ticket_embeddings=np.load(embeddings_path)

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
        mask=df_resolved['queue']==predicted_category
        filtered_df=df_resolved[mask].reset_index(drop=True)
        filtered_embeddings=ticket_embeddings[mask.values].astype('float32')

        temp_index=faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)

        distances, indices= temp_index.search(query_embedding, min(k,len(filtered_df)))
        source_df=filtered_df
        mode="filtrat"
    else:
        distances,indices=index.search(query_embedding,k)
        source_df=df_resolved
        mode="nefiltrat"

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
client_gemini=genai.Client(api_key=GEMINI_API_KEY)

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

def generate_response_with_retry(ticket_text,similar_results,max_retries=6):
    for attempt in range(max_retries):
        try:
            return generate_response(ticket_text,similar_results)
        except Exception as e:
            print(f"Incercare {attempt+1}/{max_retries} esuata: {e}")
            if attempt< max_retries-1:
                wait_time=15*(attempt+1)
                print(f"Astept {wait_time} secunde...")
                time.sleep(wait_time)
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
    category,confidence,mode, results=find_similar_tickets_hybrid(ticket_text)
    min_distance=min(r['distance'] for r in results) if results else float('inf')

    if confidence >= confidence_threshold and min_distance <= distance_threshold:
        decision="AUTO_RESOLVE"
        response=generate_response_with_retry(ticket_text,results)
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

logger= logging.getLogger('ticket_triage')
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler=logging.FileHandler('ticket_triage.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)


def route_ticket_with_logging(ticket_text,confidence_threshold=0.5, distance_threshold=0.85):
    result=route_ticket(ticket_text,confidence_threshold,distance_threshold)
    log_entry={
        'timestamp':datetime.now().isoformat(),
        'ticket':ticket_text,
        'category':result['category'],
        'confidence':round(float(result['confidence']),3),
        'retrieval_mode':result['retrieval_mode'],
        'min_distance':round(float(result['min_distance']),3),
        'decision':result['decision']
    }

    logger.info(json.dumps(log_entry))
    return result