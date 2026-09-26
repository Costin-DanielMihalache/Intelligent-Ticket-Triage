from fastapi import FastAPI
from pydantic import BaseModel
from models import route_ticket_with_logging

app=FastAPI(title="Intelligent Ticket Triage API")

class TicketRequest(BaseModel):
    text:str

@app.post("/process-ticket")
def process_ticket(request:TicketRequest):
    result=route_ticket_with_logging(request.text)
    return result