from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints
from models import route_ticket_with_logging #logging-enabled version, used in production
# (not route_ticket directly - we want every request recorded for monitoring/audit)
from typing import Annotated
from fastapi import HTTPException

app=FastAPI(title="Intelligent Ticket Triage API")

TicketText= Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=2000)]

class TicketRequest(BaseModel):
    text:TicketText

@app.post("/process-ticket")
def process_ticket(request:TicketRequest):
    try:
        result=route_ticket_with_logging(request.text)
        return result
    except Exception as e:
        # Catch any unexpected error (e.g. Gemini API down) to return a clean
        # HTTP response instead of letting the server crash
        raise HTTPException(status_code=500, detail=f"Error processing ticket: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}