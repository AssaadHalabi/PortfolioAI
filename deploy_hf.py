"""
Deploy this Gradio app to Hugging Face Spaces.

Prerequisites:
1. Create a Hugging Face write token at https://huggingface.co/settings/tokens
2. Add it to .env:
      HF_TOKEN=hf_...
3. Optional:
      HF_SPACE_ID=your-username/portfolio-qa

Run:
    python deploy_hf.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_file


ROOT = Path(__file__).resolve().parent
SPACE_SDK = "gradio"
DEFAULT_SPACE_NAME = "portfolio-qa"
FILES_TO_UPLOAD = [
    "app.py",
    "requirements.txt",
    "README.md",
]
SPACE_SECRETS = [
    "GROQ_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_NAMESPACE",
    "EMBED_MODEL_NAME",
    "GROQ_MODEL",
    "TOP_K",
]


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing {name}. Add it to .env and rerun this script.")


def main() -> None:
    load_dotenv(ROOT / ".env")

    token = required_env("HF_TOKEN")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = os.getenv("HF_SPACE_ID", f"{user}/{DEFAULT_SPACE_NAME}")

    print(f"Creating/updating Hugging Face Space: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk=SPACE_SDK,
        token=token,
        exist_ok=True,
        private=False,
    )

    for secret_name in SPACE_SECRETS:
        value = os.getenv(secret_name)
        if value is not None:
            api.add_space_secret(repo_id=repo_id, key=secret_name, value=value)
            print(f"Set Space secret: {secret_name}")

    for filename in FILES_TO_UPLOAD:
        path = ROOT / filename
        if not path.exists():
            raise FileNotFoundError(path)

        upload_file(
            repo_id=repo_id,
            repo_type="space",
            path_or_fileobj=str(path),
            path_in_repo=filename,
            token=token,
        )
        print(f"Uploaded: {filename}")

    print("\nPermanent Space URL:")
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
