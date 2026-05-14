"""
Gradio Q&A app for the Pinecone-backed portfolio RAG system.

Run with:
    py app.py

Then open:
    http://localhost:7860
"""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import gradio_client.utils as gradio_client_utils
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()


def _patch_gradio_additional_properties_bool() -> None:
    """
    gradio_client mishandles JSON Schema where additionalProperties is a boolean
    (e.g. true). It recurses into that value and calls get_type(bool), which
    crashes on `"const" in schema`. Normalize bool / non-dict leaves to "Any".
    """
    _orig = gradio_client_utils._json_schema_to_python_type

    def _wrapper(schema: Any, defs: Any) -> str:
        if isinstance(schema, bool):
            return "Any"
        if not isinstance(schema, dict):
            return "Any"
        return _orig(schema, defs)

    gradio_client_utils._json_schema_to_python_type = _wrapper


_patch_gradio_additional_properties_bool()

DEFAULT_INDEX_NAME = "portfolio-qa"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_TOP_K = 4


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


def get_match_metadata(match: Any) -> dict[str, Any]:
    metadata = getattr(match, "metadata", None)
    if metadata is None and isinstance(match, dict):
        metadata = match.get("metadata")
    return metadata or {}


def get_match_score(match: Any) -> float:
    score = getattr(match, "score", None)
    if score is None and isinstance(match, dict):
        score = match.get("score", 0.0)
    return float(score or 0.0)


def get_matches(results: Any) -> list[Any]:
    matches = getattr(results, "matches", None)
    if matches is None and isinstance(results, dict):
        matches = results.get("matches", [])
    return list(matches or [])


GROQ_API_KEY = required_env("GROQ_API_KEY")
PINECONE_API_KEY = required_env("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL)
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
TOP_K = int(os.getenv("TOP_K", str(DEFAULT_TOP_K)))


print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

stats = index.describe_index_stats()
vector_count = value_get(stats, "total_vector_count", 0)
print(f"Pinecone index ready. {vector_count} vectors loaded.")

print("Connecting to Groq...")
groq_client = Groq(api_key=GROQ_API_KEY)
print("All systems ready.\n")


SYSTEM_PROMPT = """You are an AI assistant that answers questions about Assaad El Halabi.
You answer strictly from the provided resume/profile context.

Rules:
- Only use information present in the context. Never invent or assume facts.
- Be concise and professional. Use 2-4 sentences unless more detail is needed.
- Speak in third person: "Assaad has..." not "I have..."
- If the answer is not in the context, say exactly:
  "I don't have that specific information in Assaad's profile."
- Do not apologize or pad your answers."""


def rag_answer(question: str) -> tuple[str, list[str]]:
    """
    Embed a question, retrieve the closest resume chunks from Pinecone, and
    generate a grounded answer with Groq.
    """
    question = question.strip()
    if not question:
        return "", []

    question_vector = embed_model.encode(question).tolist()
    query_args: dict[str, Any] = {
        "vector": question_vector,
        "top_k": TOP_K,
        "include_metadata": True,
    }
    if PINECONE_NAMESPACE:
        query_args["namespace"] = PINECONE_NAMESPACE

    results = index.query(**query_args)

    source_chunks: list[str] = []
    for match in get_matches(results):
        metadata = get_match_metadata(match)
        text = str(metadata.get("text", "")).strip()
        if text:
            score = get_match_score(match)
            source_chunks.append(f"[similarity: {score:.2f}] {text}")

    if not source_chunks:
        return "I don't have that specific information in Assaad's profile.", []

    context = "\n\n---\n\n".join(source_chunks)
    user_message = f"""Context from Assaad's resume and profile:

{context}

Question: {question}"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()
    return answer, source_chunks


def format_sources(source_chunks: list[str]) -> str:
    if not source_chunks:
        return "No source chunks retrieved yet."

    lines = []
    for index_number, chunk in enumerate(source_chunks, start=1):
        lines.append(f"### Source {index_number}\n{chunk}")
    return "\n\n".join(lines)


def chat(question: str, history: Any) -> tuple[str, Any, str]:
    if not question.strip():
        return "", history, "No source chunks retrieved yet."

    if history is None:
        history = []

    answer, sources = rag_answer(question)
    history.append((question, answer))
    return "", history, format_sources(sources)


SUGGESTIONS = [
    "What is Assaad's experience with LLMs?",
    "Where has Assaad worked?",
    "What is Assaad's educational background?",
    "What tech stack does Assaad use?",
    "What was Assaad's most impactful project?",
    "Is Assaad available for remote work?",
    "What languages does Assaad speak?",
    "What are Assaad's salary expectations?",
]

CSS = """
* { box-sizing: border-box; }

body,
.gradio-container {
    background:
        radial-gradient(circle at 16% 4%, rgba(200, 240, 96, 0.10), transparent 24rem),
        linear-gradient(135deg, #0a0d0c 0%, #111313 48%, #0a0a0a 100%) !important;
    color: #f3efe4 !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

html,
body,
gradio-app,
.gradio-container,
.main,
.app,
#root {
    min-width: 100% !important;
    width: 100% !important;
}

body,
gradio-app,
.main,
.app,
#root {
    background:
        radial-gradient(circle at 16% 4%, rgba(200, 240, 96, 0.10), transparent 24rem),
        linear-gradient(135deg, #0a0d0c 0%, #111313 48%, #0a0a0a 100%) !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 28px 20px 34px !important;
}

.header {
    min-height: 172px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    padding: 26px 0 24px;
    border-bottom: 1px solid rgba(243, 239, 228, 0.11);
    margin-bottom: 22px;
    text-align: center;
}

.eyebrow {
    color: #c8f060;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.76rem;
    letter-spacing: 0.12em;
    line-height: 1.5;
    margin: 0 0 10px;
    text-transform: uppercase;
}

.header h1 {
    max-width: 900px;
    color: #f6f1e7;
    font-size: clamp(2.25rem, 7vw, 5.25rem);
    line-height: 0.95;
    font-weight: 850;
    letter-spacing: 0;
    margin: 0;
}

.header p:not(.eyebrow) {
    max-width: 690px;
    color: #b8b1a3;
    font-size: 1rem;
    line-height: 1.65;
    margin: 18px 0 0;
}

.accent { color: #c8f060; }

.meta-strip {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    margin-top: 22px;
}

.meta-pill {
    border: 1px solid rgba(243, 239, 228, 0.14);
    border-radius: 999px;
    color: #d9d2c3;
    background: rgba(255, 255, 255, 0.035);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.74rem;
    letter-spacing: 0.04em;
    padding: 7px 10px;
}

.app-shell {
    align-items: stretch !important;
    gap: 18px !important;
}

.chat-panel,
.side-panel {
    border: 1px solid rgba(243, 239, 228, 0.11);
    border-radius: 8px;
    background: rgba(12, 14, 13, 0.82);
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.26);
    backdrop-filter: blur(18px);
}

.chat-panel { padding: 14px; }
.side-panel { padding: 18px; }

.chatbot {
    background: #0d100f !important;
    border: 1px solid rgba(243, 239, 228, 0.10) !important;
    border-radius: 8px !important;
    min-height: 520px;
}

.chatbot .message {
    border-radius: 8px !important;
    font-size: 0.98rem !important;
    line-height: 1.55 !important;
}

.chatbot,
.chatbot *,
.chatbot .message,
.chatbot .message *,
.chatbot .prose,
.chatbot .prose *,
.chatbot .md,
.chatbot .md * {
    color: #f6f1e7 !important;
}

.chatbot .user {
    background: #c8f060 !important;
    color: #14170e !important;
}

.chatbot .user,
.chatbot .user *,
.chatbot [data-testid="user"] *,
.chatbot .user-message *,
.chatbot .message-wrap.user * {
    color: #14170e !important;
}

.chatbot .bot {
    background: #191d1a !important;
    color: #f4efe5 !important;
    border: 1px solid rgba(243, 239, 228, 0.08) !important;
}

.chatbot .bot,
.chatbot .bot *,
.chatbot [data-testid="bot"] *,
.chatbot .bot-message *,
.chatbot .message-wrap.bot * {
    color: #f6f1e7 !important;
}

.input-row {
    gap: 10px !important;
    margin-top: 12px !important;
}

.input-row textarea {
    background: #121513 !important;
    border: 1px solid rgba(243, 239, 228, 0.12) !important;
    border-radius: 8px !important;
    color: #f6f1e7 !important;
    min-height: 50px !important;
    box-shadow: none !important;
}

.input-row textarea:focus {
    border-color: rgba(200, 240, 96, 0.72) !important;
    box-shadow: 0 0 0 3px rgba(200, 240, 96, 0.12) !important;
}

.ask-btn {
    min-height: 50px !important;
    border-radius: 8px !important;
    background: #c8f060 !important;
    border: 1px solid #c8f060 !important;
    color: #10140a !important;
    font-weight: 750 !important;
}

.side-title {
    color: #f6f1e7;
    font-size: 0.82rem;
    font-weight: 750;
    letter-spacing: 0.08em;
    margin: 0 0 12px;
    text-transform: uppercase;
}

.sug-btn {
    justify-content: flex-start !important;
    width: 100% !important;
    min-height: 40px !important;
    padding: 8px 10px !important;
    background: transparent !important;
    border: 1px solid rgba(243, 239, 228, 0.12) !important;
    border-radius: 8px !important;
    color: #d8d1c4 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-align: left !important;
}

.sug-btn:hover {
    background: rgba(200, 240, 96, 0.08) !important;
    border-color: rgba(200, 240, 96, 0.65) !important;
    color: #c8f060 !important;
}

.source-box {
    margin-top: 16px;
}

.source-box,
.source-box > *,
.source-box details,
.source-box summary,
.source-box .accordion,
.source-box .block,
.source-box .form,
.source-box .wrap,
.source-box .wrap-inner {
    background: #111513 !important;
    border-color: rgba(243, 239, 228, 0.14) !important;
    color: #f6f1e7 !important;
}

.source-box .label-wrap span {
    color: #f6f1e7 !important;
    font-size: 0.82rem !important;
    font-weight: 750 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

.source-box .prose,
.source-box .prose *,
.source-box .md,
.source-box .md *,
.source-box .markdown,
.source-box .markdown *,
.source-box .markdown-body,
.source-box .markdown-body *,
.source-box .output-markdown,
.source-box .output-markdown *,
.source-box [data-testid="markdown"],
.source-box [data-testid="markdown"] *,
.source-box [data-testid="block-info"],
.source-box [data-testid="block-info"] *,
.source-box .block-info,
.source-box .block-info *,
.source-box div,
.source-box span,
.source-box p,
.source-box li,
.source-box h1,
.source-box h2,
.source-box h3,
.source-box strong,
.source-box em {
    color: #f6f1e7 !important;
    font-size: 0.84rem !important;
    line-height: 1.55 !important;
}

.source-box h1,
.source-box h2,
.source-box h3,
.source-box strong {
    color: #c8f060 !important;
}

.source-box code,
.source-box pre,
.source-box pre * {
    color: #10140a !important;
    background: #c8f060 !important;
}

footer { display: none !important; }

@media (max-width: 860px) {
    .gradio-container {
        padding: 18px 12px 24px !important;
    }

    .header {
        min-height: 136px;
    }

    .app-shell {
        flex-direction: column !important;
    }

    .chatbot {
        min-height: 440px;
    }
}
"""


with gr.Blocks(css=CSS, title="Assaad El Halabi - Portfolio Q&A") as demo:
    gr.HTML(
        """
        <div class="header">
          <p class="eyebrow">RAG portfolio assistant</p>
          <h1>Ask Assaad's profile like a hiring conversation<span class="accent">.</span></h1>
          <p>Grounded answers from resume context, retrieved through Pinecone and answered with Groq.</p>
          <div class="meta-strip">
            <span class="meta-pill">Pinecone retrieval</span>
            <span class="meta-pill">Groq Llama 3.1</span>
            <span class="meta-pill">Source-backed answers</span>
          </div>
        </div>
        """
    )

    with gr.Row(elem_classes=["app-shell"]):
        with gr.Column(scale=8, elem_classes=["chat-panel"]):
            chatbot = gr.Chatbot(
                value=[],
                height=520,
                elem_classes=["chatbot"],
                show_label=False,
                bubble_full_width=False,
            )

            with gr.Row(elem_classes=["input-row"]):
                txt = gr.Textbox(
                    placeholder="Ask about Assaad's experience, skills, projects, or availability...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button(
                    "Ask",
                    scale=1,
                    variant="primary",
                    elem_classes=["ask-btn"],
                )

        with gr.Column(scale=4, elem_classes=["side-panel"]):
            gr.HTML("<p class='side-title'>Suggested questions</p>")

            for suggestion in SUGGESTIONS:
                btn = gr.Button(suggestion, elem_classes=["sug-btn"])
                btn.click(fn=lambda value=suggestion: value, outputs=txt, show_api=False)

            with gr.Accordion(
                "Retrieved source chunks",
                open=False,
                elem_classes=["source-box"],
            ):
                sources_md = gr.Markdown("No source chunks retrieved yet.")

    txt.submit(chat, [txt, chatbot], [txt, chatbot, sources_md], show_api=False)
    send_btn.click(chat, [txt, chatbot], [txt, chatbot, sources_md], show_api=False)


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_api=False,
    )
