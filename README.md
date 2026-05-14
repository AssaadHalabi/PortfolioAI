---
title: Assaad Portfolio QA
emoji: 💬
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# RAG Portfolio Q&A - Pinecone Version

This is a Gradio app that answers questions about a portfolio/resume using a Pinecone vector index and Groq-hosted Llama 3.

## Architecture

1. `ingest.py` reads `resume_data.txt`, splits it into overlapping chunks, embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`, and upserts the vectors plus source text into Pinecone.
2. `app.py` embeds each user question with the same model, retrieves the top matching chunks from Pinecone, and sends only those chunks plus the question to Groq for a grounded answer.

## Files

```text
.env.example       Example environment variables
.gitignore         Keeps secrets and local env files out of git
app.py             Gradio Q&A app
ingest.py          One-time Pinecone ingestion script
requirements.txt   Python dependencies
resume_data.txt    Your resume/profile text
```

## Setup on Windows

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add your API keys.

## Setup on macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your API keys.

## Environment Variables

Required:

```text
GROQ_API_KEY=your_groq_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

Optional defaults:

```text
PINECONE_INDEX_NAME=portfolio-qa
PINECONE_NAMESPACE=
CLEAR_INDEX_ON_INGEST=true
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
GROQ_MODEL=llama-3.1-8b-instant
TOP_K=4
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false
```

## Run

Replace the placeholder text in `resume_data.txt` with the actual resume/profile content, then ingest once:

```powershell
py ingest.py
```

Start the app whenever you want to use it:

```powershell
py app.py
```

Open `http://localhost:7860`.

## Deploy Permanently

For a link that lasts longer than Gradio's 72-hour temporary share URLs, deploy
to Hugging Face Spaces:

1. Create a Hugging Face write token at `https://huggingface.co/settings/tokens`.
2. Add it to `.env`:

```text
HF_TOKEN=hf_your_token_here
HF_SPACE_ID=your-username/portfolio-qa
```

`HF_SPACE_ID` is optional; if omitted, `deploy_hf.py` uses
`your-username/portfolio-qa`.

Then run:

```powershell
python deploy_hf.py
```

The script creates or updates the Space, uploads `app.py`, `requirements.txt`,
and `README.md`, and stores your Groq/Pinecone values as private Space secrets.

## Keep the Hugging Face Space Awake

This repo includes a GitHub Actions workflow at
`.github/workflows/keep-hf-space-awake.yml` that opens the deployed Space once a
day with `agent-browser`, waits for the app UI to render, captures a screenshot,
and optionally posts the result to Discord.

In your GitHub repository, add these Actions secrets:

```text
HUGGING_FACE_SPACE_URL=https://huggingface.co/spaces/your-username/portfolio-qa
DISCORD_WEBHOOK_URL=your_discord_channel_webhook_url
```

`DISCORD_WEBHOOK_URL` is optional. If it is not set, the workflow still checks
the Space and uploads the screenshot as a GitHub Actions artifact.

The workflow waits for `Ask Assaad` by default. To use different readiness text,
add a repository variable named `HUGGING_FACE_READY_TEXT`.

If you previously installed dependencies before the FastAPI/Starlette pins were
added, run this once inside the activated virtual environment:

```powershell
py -m pip install --upgrade --force-reinstall -r requirements.txt
```

## Pinecone Notes

The ingestion script creates the Pinecone index automatically if it does not exist. If you create it manually, use:

```text
Name: portfolio-qa
Dimensions: 384
Metric: cosine
Cloud/region: aws/us-east-1
```

By default, ingestion clears the configured namespace before upserting. This keeps stale chunks from old resume versions out of search results.
