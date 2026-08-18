"""Preload and cache heavy ML models used by the project.

This script will instantiate the embedding model once and run a dummy
embedding to ensure Hugging Face weights are downloaded and cached, avoiding
on-startup downloads when the app runs.

Set `HF_TOKEN` in your environment for faster, authenticated downloads.
"""
import os
from app.config.settings import settings

def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set — downloads may be rate-limited.")

    print(f"Preloading embedding model: {settings.embedding_model}")

    # Import locally to avoid requiring ML deps when not using this script
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    # Run a single embedding to force model download and cache
    print("Running dummy embedding to download weights...")
    _ = emb.embed_query("warmup cache")
    print("Model downloaded and cached.")


if __name__ == "__main__":
    main()
