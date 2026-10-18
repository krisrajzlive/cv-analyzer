# CV Analyzer

A Flask web app that scores a resume against a job description and tells the candidate how to improve it. It combines three Azure services: Document Intelligence for parsing, Blob Storage for keeping qualified resumes, and a Microsoft Foundry model deployment for written feedback.

## How it works

```
Upload CV + job description
        │
        ▼
Azure Document Intelligence (prebuilt-layout)  ──► extracted text, cleaned and split into sections
        │
        ▼
TF-IDF cosine similarity vs. job description   ──► match score (0–100%)
        │
        ├── score ≥ MATCH_THRESHOLD ──► "Qualified"     ──► resume uploaded to Azure Blob Storage
        └── score <  MATCH_THRESHOLD ──► "Not Qualified" ──► resume not stored
        │
        ▼
Foundry model (Azure OpenAI deployment)         ──► four actionable improvement suggestions
        │
        ▼
Result page: score, status, feedback, extracted text
```

The uploaded file is deleted from local disk after each analysis; only qualified resumes are kept, in Blob Storage.

## Azure services used

| Service | Role | What you need |
|---|---|---|
| **Azure AI Document Intelligence** | OCR and layout extraction (`prebuilt-layout`) for PDF, DOCX, PNG, JPG | A Document Intelligence (or multi-service AI) resource: endpoint + key |
| **Azure Storage account (Blob)** | Stores resumes that meet the match threshold | Storage account connection string. The container is created automatically if missing |
| **Microsoft Foundry / Azure OpenAI** | Generates resume feedback with a chat model deployment (tested with `gpt-5-nano`) | Foundry resource endpoint + key, and the name of a deployed chat model |

## Configuration (`.env`)

Copy `.env.example` to `.env` and fill in the values. `.env` is git-ignored; never commit it.

| Variable | Required | Description |
|---|---|---|
| `AZURE_ENDPOINT` | yes | Document Intelligence endpoint, e.g. `https://<resource>.cognitiveservices.azure.com/` |
| `AZURE_KEY` | yes | Document Intelligence key (Azure portal → resource → Keys and Endpoint) |
| `AZURE_STORAGE_CONNECTION_STRING` | yes | Storage account → Security + networking → Access keys → Connection string |
| `AZURE_BLOB_CONTAINER_NAME` | no | Container for qualified resumes (default `qualified-resumes`) |
| `AZURE_OPENAI_ENDPOINT` | for feedback | Foundry/Azure OpenAI endpoint, e.g. `https://<resource>.cognitiveservices.azure.com/` |
| `AZURE_OPENAI_KEY` | for feedback | Key for the Foundry/Azure OpenAI resource |
| `AZURE_OPENAI_DEPLOYMENT` | for feedback | Name of the deployed chat model (e.g. `gpt-5-nano`) |
| `AZURE_OPENAI_API_VERSION` | no | API version, default `2024-12-01-preview`. This is an API version, not the model version |
| `MATCH_THRESHOLD` | no | Cosine-similarity cutoff for "Qualified" (default `0.65`) |

If the Foundry settings are missing the app still runs; the feedback section reports that it is unavailable.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

copy .env.example .env          # Windows  (cp .env.example .env on macOS / Linux)
# edit .env and fill in your endpoints and keys

python app.py
```

Open http://127.0.0.1:5000, upload a resume (PDF, DOCX, PNG, JPG or JPEG, max 5 MB), paste the job description and click **Analyze Resume**.

## Project layout

```
app.py              Flask app: parsing, scoring, feedback, blob upload
templates/
  upload.html       Upload form
  result.html       Analysis results
requirements.txt
.env.example        Configuration template
```

## Notes

- The match score is TF-IDF cosine similarity: it measures keyword overlap, not meaning. Typical real-world scores are well below 100%, so tune `MATCH_THRESHOLD` for your use.
- Resume text and the job description are sent to the Foundry model to generate feedback.
- Runs with Flask's development server; use a production WSGI server (e.g. gunicorn/waitress) and authentication before exposing it publicly.
