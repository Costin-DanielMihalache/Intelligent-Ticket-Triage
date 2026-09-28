import os
os.environ.setdefault("GEMINI_API_KEY","test")

import pytest

@pytest.fixture(autouse=True)
def mock_gemini(monkeypatch):
    monkeypatch.setattr(
        "models.generate_response_with_retry",
        lambda *a, **k: "Mock Response from Gemini",
    )