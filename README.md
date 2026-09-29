# Intelligent-Ticket-Triage

API that classifies incoming support tickets with a fine-tuned DistilBERT model, retrieves similar previously-resolved tickets using sentence embeddings and FAISS, and drafts a reply with Gemini. Tickets the system isn't confident about are escalated to a human instead of being auto-answered.

**Live demo:** [https://intelligent-ticket-triage-git-746339824444.europe-west1.run.app/docs](https://intelligent-ticket-triage-git-746339824444.europe-west1.run.app/docs)

## How it works

```
Ticket text
  -> DistilBERT classifier          (predicted category + confidence)
  -> FAISS similarity search        (most similar resolved tickets, filtered by category if confidence is high)
  -> Gemini                         (drafts a reply using the retrieved examples as context)
  -> decision: AUTO_RESOLVE or ESCALATE_TO_HUMAN
```

A ticket is only auto-resolved if **both** the classifier is confident (confidence >= 0.5) **and** a sufficiently similar past ticket was found (distance <= 0.85). Either condition failing sends the ticket to a human. These thresholds were chosen empirically by observing scores on test examples, not through a formal search — a natural next step for calibration.

If Gemini fails after retries, or if it returns a reply containing an unfilled placeholder (e.g. `<tel_num>`, `[Your Name]` — leftover from the anonymized training examples), the ticket is escalated instead of sending a broken or empty reply to the customer.

## Stack

- **API:** FastAPI
- **Classification:** DistilBERT, fine-tuned on a labeled ticket dataset ([model on Hugging Face](https://huggingface.co/mihalachecostindaniel/ticket-classifier-distilbert))
- **Retrieval:** sentence-transformers (all-MiniLM-L6-v2) + FAISS
- **Generation:** Google Gemini
- **Deployment:** Docker, Google Cloud Run, deployed automatically on push via Cloud Build

## Running locally

```bash
git clone https://github.com/Costin-DanielMihalache/Intelligent-Ticket-Triage.git
cd Intelligent-Ticket-Triage/api
pip install -r requirements.txt
```

Create `api/.env` from `api/.env.example` and add your `GEMINI_API_KEY`.

```bash
uvicorn main:app --reload
```

Swagger UI is at `http://localhost:8000/docs`.

## Example request

```bash
curl -X POST http://localhost:8000/process-ticket \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for the same order, please refund the duplicate charge"}'
```

Response:

```json
{
  "ticket": "I was charged twice for the same order, please refund the duplicate charge",
  "category": "Billing and Payments",
  "confidence": 0.94,
  "retrieval_mode": "filtered",
  "min_distance": 0.31,
  "decision": "AUTO_RESOLVE",
  "generated_response": "Thank you for reaching out about the duplicate charge..."
}
```

## Evaluation

**Classification:** evaluated on a held-out test split (20%, stratified by category, 2,385 tickets). Overall accuracy: **41.8%** across 10 categories. Accuracy on tickets the system actually auto-resolves (confidence >= 0.5, distance <= 0.85) is higher, **59.2%**, versus 30.1% on escalated ones — confirming the thresholds meaningfully separate reliable from unreliable predictions. Note: the test split was also used for best-checkpoint selection during fine-tuning, so these numbers are slightly optimistic. Interestingly, plain 1-NN retrieval on the sentence embeddings alone reaches 67.9% accuracy on the same test set, outperforming the DistilBERT classifier — a sign retrieval carries more signal here than classification.

**Generation:** ran a small manual eval (`api/eval_generation.py` + `api/eval_summarize.py`) — samples tickets stratified by category, runs the full pipeline, and writes results to a CSV for manual grading. On a sample of 8 `AUTO_RESOLVE` tickets:
- Mean relevance: 3.5 / 5
- Hallucinations (invented facts): 0 / 8
- Found that 6/8 replies contained an unfilled placeholder copied from the retrieved examples — fixed by detecting and escalating such replies instead of sending them (see `contains_unfilled_placeholder` in `api/models.py`)

## Known limitations

- Classification accuracy is limited (41.8% across 10 categories); 1-NN retrieval alone outperforms it (67.9%), suggesting the classifier could be dropped or replaced by a retrieval-based category vote
- Generation eval sample is small (8 graded replies); not a statistically rigorous evaluation
- The `/process-ticket` endpoint has no authentication or rate limiting — anyone with the URL can consume the Gemini quota
- No load/concurrency testing has been done
- Thresholds (confidence, distance) were chosen empirically, not tuned

## Possible next steps

- Add API key auth / rate limiting on the public endpoint
- Expand the eval set and calibrate the confidence/distance thresholds against it
- Pin the Hugging Face model to a specific revision instead of always pulling `main`