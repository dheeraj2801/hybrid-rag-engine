from uuid import uuid4

from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.vector_store import vector_store


def main():
    documents = load_documents("data")

    chunks = chunk_documents(documents)

    # Qdrant requires point IDs to be integers or UUID strings.
    # Generate UUIDs for storage and keep the original chunk id in metadata.
    ids = [str(uuid4()) for _ in chunks]

    metadatas = [
        {
            "chunk_id": chunk["id"],
            "source": chunk["source"],
            "parent_id": chunk["parent_id"],
            "orig_id": chunk["id"],
        }
        for chunk in chunks
    ]

    vector_store.add_texts(
        texts=[chunk["text"] for chunk in chunks],
        metadatas=metadatas,
        ids=ids,
    )

    print(f"Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    main()