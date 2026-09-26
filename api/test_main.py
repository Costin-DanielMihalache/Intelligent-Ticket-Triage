from fastapi.testclient import TestClient
from main import app

client=TestClient(app)

def test_health_check():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()=={"status":"ok"}

def test_process_ticket_clear_case():
    response=client.post("/process-ticket",json={"text": "My card was charged twice for the same order"})
    assert response.status_code == 200
    data=response.json()
    assert "category" in data
    assert "decision" in data
    assert data["decision"] in ["AUTO_RESOLVE","ESCALATE_TO_HUMAN"]

def test_process_ticket_vague_input():
    response=client.post("/process-ticket",json={"text":"asdf random text"})
    assert response.status_code == 200
    data= response.json()
    assert data["decision"]=="ESCALATE_TO_HUMAN"

def test_process_ticket_empty_text():
    response=client.post("/process-ticket", json={"text":""})
    assert response.status_code==200