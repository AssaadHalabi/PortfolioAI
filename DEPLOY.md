# Deployment Notes - Pinecone RAG Version

This project can run locally as-is. For a public portfolio deployment, use a host that supports:

- A Python Gradio web app
- Outbound API calls to Pinecone and Groq
- Persistent environment secrets

Hugging Face Spaces, Render, Railway, and Fly.io can all work. The easiest path is usually Hugging Face Spaces with the Gradio SDK.

## Required Secrets

Add these as private secrets in your deployment provider:

```text
GROQ_API_KEY=your_groq_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=portfolio-qa
PINECONE_NAMESPACE=
```

Optional settings:

```text
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
GROQ_MODEL=llama-3.1-8b-instant
TOP_K=4
```

## Important Deployment Flow

Run ingestion before deploying the app:

```powershell
py ingest.py
```

That uploads your resume chunks to Pinecone. After the vectors are in Pinecone, the deployed app only needs to run `app.py`; it does not need to re-ingest on startup.

## Hugging Face Spaces

The automated route:

1. Create a write token at `https://huggingface.co/settings/tokens`.
2. Add it to `.env` as `HF_TOKEN=hf_...`.
3. Optionally add `HF_SPACE_ID=your-username/portfolio-qa`.
4. Run `python deploy_hf.py`.

The script creates or updates the Space, uploads the app files, and adds the
required Groq/Pinecone values as private Space secrets.

The manual route:

1. Create a new Space at `https://huggingface.co/new-space`.
2. Choose SDK: `Gradio`.
3. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. Add the secrets listed above in Space settings.
5. Let the Space build and launch.

Do not upload `.env`.

## How RAG Works Here

1. Ingestion: `resume_data.txt` is split into about 250-word chunks with overlap.
2. Embedding: each chunk is converted into a 384-dimensional vector using `all-MiniLM-L6-v2`.
3. Storage: vectors and chunk text are stored in Pinecone.
4. Retrieval: each user question is embedded and Pinecone returns the top matching chunks.
5. Generation: Groq Llama 3 receives the retrieved chunks and answers only from that context.

The retrieval store is Pinecone, not ChromaDB, so the knowledge base persists independently of the running app.
