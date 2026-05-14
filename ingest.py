"""
One-time ingestion script for the Pinecone-backed portfolio RAG system.

This script:
1. Reads resume_data.txt
2. Splits it into overlapping chunks
3. Embeds each chunk with sentence-transformers
4. Creates a Pinecone index if needed
5. Upserts the chunks and their metadata into Pinecone

Run with:
    py ingest.py
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
RESUME_PATH = ROOT / "resume_data.txt"

DEFAULT_INDEX_NAME = "portfolio-qa"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 40


def value_get(value: Any, key: str, default: Any = None) -> Any:
    if hasattr(value, "get"):
        return value.get(key, default)
    return getattr(value, key, default)


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(
        f"Missing {name}. Create a .env file from .env.example and set this value."
    )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-bounded chunks while trying to keep
    sentences intact.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = [s.strip() for s in re.split(
        r"(?<=[.!?\n])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if word_count > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_length = 0
            chunks.extend(split_long_sentence(words, chunk_size, overlap))
            continue

        if current_sentences and current_length + word_count > chunk_size:
            chunks.append(" ".join(current_sentences))
            current_sentences, current_length = overlap_tail(
                current_sentences, overlap)

        current_sentences.append(sentence)
        current_length += word_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def split_long_sentence(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        window = words[start: start + chunk_size]
        if window:
            chunks.append(" ".join(window))
    return chunks


def overlap_tail(sentences: list[str], overlap: int) -> tuple[list[str], int]:
    tail: list[str] = []
    word_count = 0

    for sentence in reversed(sentences):
        sentence_words = len(sentence.split())
        if word_count + sentence_words > overlap:
            break
        tail.insert(0, sentence)
        word_count += sentence_words

    return tail, word_count


def list_index_names(pc: Pinecone) -> list[str]:
    indexes = pc.list_indexes()
    if hasattr(indexes, "names"):
        return list(indexes.names())

    names: list[str] = []
    for index in indexes:
        if isinstance(index, dict):
            names.append(index["name"])
        else:
            names.append(index.name)
    return names


def status_ready(status: Any) -> bool:
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


def description_status(description: Any) -> Any:
    if isinstance(description, dict):
        return description.get("status", {})
    return getattr(description, "status", {})


def wait_until_ready(pc: Pinecone, index_name: str, timeout_seconds: int = 120) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        description = pc.describe_index(index_name)
        if status_ready(description_status(description)):
            return
        print("Waiting for Pinecone index to be ready...")
        time.sleep(5)

    raise TimeoutError(
        f"Pinecone index '{index_name}' was not ready after {timeout_seconds}s.")


def main() -> None:
    load_dotenv()

    pinecone_api_key = required_env("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    namespace = os.getenv("PINECONE_NAMESPACE", "")
    clear_index = os.getenv("CLEAR_INDEX_ON_INGEST", "true").lower() == "true"
    embed_model_name = os.getenv("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL)
    pinecone_cloud = os.getenv("PINECONE_CLOUD", DEFAULT_CLOUD)
    pinecone_region = os.getenv("PINECONE_REGION", DEFAULT_REGION)

    if not RESUME_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RESUME_PATH}. Add your resume/profile text there before ingesting."
        )

    print("Loading resume text...")
    text = RESUME_PATH.read_text(encoding="utf-8").strip()
    if not text or "Replace this file" in text:
        raise RuntimeError(
            "resume_data.txt still contains placeholder text. Replace it with your resume/profile content."
        )
    print(f"Loaded {len(text)} characters of resume text.")

    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError("No chunks were generated from resume_data.txt.")

    print(f"Split into {len(chunks)} chunks.")
    for index_number, chunk in enumerate(chunks):
        preview = chunk.replace("\n", " ")[:80]
        print(
            f"  Chunk {index_number}: {len(chunk.split())} words - {preview}...")

    print(f"\nLoading embedding model ({embed_model_name})...")
    model = SentenceTransformer(embed_model_name)

    print("Embedding chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embedding_dimension = len(embeddings[0])
    print(
        f"Generated {len(embeddings)} embeddings with {embedding_dimension} dimensions.")

    print("\nConnecting to Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)

    if index_name not in list_index_names(pc):
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region),
        )
    else:
        print(f"Pinecone index '{index_name}' already exists.")

    wait_until_ready(pc, index_name)
    index = pc.Index(index_name)

    if clear_index:
        print("Clearing existing vectors before upsert...")
        delete_args: dict[str, Any] = {"delete_all": True}
        if namespace:
            delete_args["namespace"] = namespace
        try:
            index.delete(**delete_args)
        except NotFoundException:
            # New index or namespace that never had vectors — nothing to delete.
            pass

    vectors = []
    for index_number, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append(
            {
                "id": f"chunk_{index_number}",
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk,
                    "chunk_index": index_number,
                    "source": "resume_data.txt",
                },
            }
        )

    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start: start + batch_size]
        upsert_args: dict[str, Any] = {"vectors": batch}
        if namespace:
            upsert_args["namespace"] = namespace
        index.upsert(**upsert_args)
        print(
            f"Upserted batch {start // batch_size + 1} ({len(batch)} vectors).")

    stats = index.describe_index_stats()
    print(
        f"\nDone. Pinecone index now contains {value_get(stats, 'total_vector_count', 0)} vectors.")
    print("You can now run app.py to start the Q&A interface.")


if __name__ == "__main__":
    main()
