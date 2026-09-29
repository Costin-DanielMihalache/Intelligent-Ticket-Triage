import pytest
import models

real_retry=models.generate_response_with_retry  # captured before conftest's autouse mock applies

class FakeAPIError(Exception):
    def __init__(self,code):
        super().__init__(f"HTTP {code}")
        self.code=code

@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps=[]
    monkeypatch.setattr(models.time,"sleep",lambda s: sleeps.append(s))
    return sleeps

def _failing(monkeypatch,code,fail_times=None):
    calls={"n":0}
    def fake(*a,**k):
        calls["n"]+=1
        if fail_times is None or calls["n"]<=fail_times:
            raise FakeAPIError(code)
        return "ok"
    monkeypatch.setattr(models,"generate_response",fake)
    return calls

def test_retry_recovers_after_transient_errors(monkeypatch,no_sleep):
    calls=_failing(monkeypatch,503,fail_times=2)
    assert real_retry("t",[])=="ok"
    assert calls["n"]==3
    assert len(no_sleep)==2  # waited between attempts, not after success

def test_retry_gives_up_after_max_retries(monkeypatch):
    calls=_failing(monkeypatch,503)
    assert real_retry("t",[],max_retries=3) is None
    assert calls["n"]==3

def test_retry_does_not_retry_client_errors(monkeypatch):
    calls=_failing(monkeypatch,401)
    assert real_retry("t",[]) is None
    assert calls["n"]==1

def test_backoff_is_bounded(monkeypatch,no_sleep):
    _failing(monkeypatch,503)
    real_retry("t",[],max_retries=5,base_delay=1.0,max_delay=8.0)
    assert len(no_sleep)==4
    assert all(0<=s<=min(8.0,2**i) for i,s in enumerate(no_sleep))

def _confident_match(monkeypatch):
    monkeypatch.setattr(models,"find_similar_tickets_hybrid",
        lambda text:("Billing",0.9,"filtered",[{"subject":"s","queue":"Billing","answer":"a","distance":0.1}]))

def test_escalates_when_gemini_fails(monkeypatch):
    _confident_match(monkeypatch)
    monkeypatch.setattr(models,"generate_response_with_retry",lambda *a,**k: None)
    result=models.route_ticket("card charged twice")
    assert result["decision"]=="ESCALATE_TO_HUMAN"
    assert result["generated_response"] is None

def test_auto_resolves_when_gemini_answers(monkeypatch):
    _confident_match(monkeypatch)
    result=models.route_ticket("card charged twice")
    assert result["decision"]=="AUTO_RESOLVE"

def test_escalates_when_response_has_unfilled_placeholder(monkeypatch):
    _confident_match(monkeypatch)
    monkeypatch.setattr(models,"generate_response_with_retry",
        lambda *a,**k: "Thanks for reaching out. Call me at <tel_num>.")
    result=models.route_ticket("card charged twice")
    assert result["decision"]=="ESCALATE_TO_HUMAN"
    assert result["generated_response"] is None

def test_does_not_escalate_on_clean_response(monkeypatch):
    _confident_match(monkeypatch)
    monkeypatch.setattr(models,"generate_response_with_retry",
        lambda *a,**k: "Thanks for reaching out. We'll look into it.")
    result=models.route_ticket("card charged twice")
    assert result["decision"]=="AUTO_RESOLVE"

def test_contains_unfilled_placeholder():
    assert models.contains_unfilled_placeholder("Call me at <tel_num>.")
    assert models.contains_unfilled_placeholder("Reach out to [Your Name].")
    assert not models.contains_unfilled_placeholder("This is a normal reply.")