from fastapi import FastAPI
from pydantic import BaseModel
from models import route_ticket_with_logging
from fastapi import HTTPException

app=FastAPI(title="Intelligent Ticket Triage API")

class TicketRequest(BaseModel):
    text:str

@app.post("/process-ticket")
def process_ticket(request:TicketRequest):
    try:
        result=route_ticket_with_logging(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ticket: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}