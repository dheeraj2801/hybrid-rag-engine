"""Utility to initialize Qdrant collection used by the project.

This script avoids importing `app.retrieval.vector_store` to prevent import-
time instantiation of `QdrantVectorStore`. It uses a standalone `QdrantClient`
to create/delete collections.
"""
import argparse

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config.settings import settings


def main(force: bool = False):
    client = QdrantClient(url=settings.qdrant_url)
    name = settings.qdrant_collection
    size = settings.vector_size

    try:
        info = client.get_collection(name)
        # robustly extract existing vector size from known response shapes
        existing_size = None
        # preferred: object attribute path returned by qdrant_client
        try:
            existing_size = info.config.params.vectors.size
        except Exception:
            pass

        # object-like response may expose `vectors_config` on some versions
        if existing_size is None:
            try:
                existing_size = getattr(info, "vectors_config").size
            except Exception:
                pass

        # dict-like responses may nest under result.config.params.vectors
        if existing_size is None:
            try:
                existing_size = info["result"]["config"]["params"]["vectors"]["size"]
            except Exception:
                pass

        # fallback older shape: result.config.vectors.size
        if existing_size is None:
            try:
                existing_size = info["result"]["config"]["vectors"]["size"]
            except Exception:
                existing_size = None

        if existing_size is None:
            print(f"Collection '{name}' exists but vector size couldn't be read; leaving as-is")
            return

        if int(existing_size) != int(size):
            msg = (
                f"Existing collection '{name}' uses vectors size={existing_size}, requested size={size}."
            )
            if not force:
                print(msg + " Use --force to recreate the collection.")
                return
            # recreate collection
            client.delete_collection(collection_name=name)
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
            print(f"Recreated collection '{name}' with size={size}")
            return

        print(f"Collection '{name}' already exists and is compatible (size={size})")

    except UnexpectedResponse:
        # collection missing; create it
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        print(f"Created collection '{name}' (size={size})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Recreate collection if dimensions mismatch"
    )
    args = parser.parse_args()
    main(force=args.force)
