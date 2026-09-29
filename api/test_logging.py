import json
import logging
import models

class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records=[]
    def emit(self,record):
        self.records.append(record)

def _log_one_ticket(monkeypatch,text):
    monkeypatch.setattr(models,"find_similar_tickets_hybrid",
        lambda t:("Billing",0.9,"filtered",[{"subject":"s","queue":"Billing","answer":"a","distance":0.1}]))
    handler=Capture()
    models.logger.addHandler(handler)
    try:
        models.route_ticket_with_logging(text)
    finally:
        models.logger.removeHandler(handler)
    return handler.records[0]

def test_log_does_not_contain_ticket_text(monkeypatch):
    text="my password is hunter2"
    record=_log_one_ticket(monkeypatch,text)
    line=models.JsonFormatter().format(record)
    assert text not in line
    entry=json.loads(line)
    assert entry["ticket_length"]==len(text)
    assert entry["decision"]=="AUTO_RESOLVE"

def test_log_line_is_structured_json(monkeypatch):
    record=_log_one_ticket(monkeypatch,"card charged twice")
    entry=json.loads(models.JsonFormatter().format(record))
    assert entry["severity"]=="INFO"
    assert entry["message"]=="ticket_routed"
    assert len(entry["ticket_hash"])==12