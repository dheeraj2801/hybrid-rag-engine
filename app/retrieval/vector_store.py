from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config.settings import settings
from app.llm.embeddings import embeddings


client = QdrantClient(
    url=settings.qdrant_url,
)


def ensure_collection(
    collection_name: str | None = None,
    vector_size: int | None = None,
    distance: str = "Cosine",
    force_recreate: bool = False,
):
    """Ensure a Qdrant collection exists. Creates it if missing.

    If the collection exists but has a different vector size, and
    `force_recreate` is True the collection will be deleted and recreated.
    Returns True if a collection was created (new or recreated), False if it
    already existed and no action was needed.
    """
    name = collection_name or settings.qdrant_collection
    size = vector_size or settings.vector_size

    try:
        info = client.get_collection(name)
        # try to extract existing vector size robustly
        existing_size = None
        try:
            existing_size = getattr(info, "vectors_config").size
        except Exception:
            try:
                # info might be a dict-like response
                existing_size = (
                    info["result"]["config"]["vectors"]["size"]
                )
            except Exception:
                existing_size = None

        if existing_size is None:
            return False

        if int(existing_size) != int(size):
            msg = (
                f"Existing collection '{name}' uses vectors size={existing_size}, "
                f"requested size={size}."
            )
            if not force_recreate:
                raise RuntimeError(
                    msg + " Set force_recreate=True to recreate the collection."
                )
            # recreate collection
            client.delete_collection(collection_name=name)
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
            return True

        return False

    except UnexpectedResponse:
        # collection missing; create it
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        return True


# Ensure collection exists before instantiating the vector store. This avoids
# langchain_qdrant validating the collection and raising a mismatch error at
# import time. If the sizes mismatch the app will recreate the collection only
# when `settings.force_recreate_qdrant` is True.
ensure_collection(force_recreate=settings.force_recreate_qdrant)


vector_store = QdrantVectorStore(
    client=client,
    collection_name=settings.qdrant_collection,
    embedding=embeddings,
)
