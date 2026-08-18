from pathlib import Path


def load_documents(directory: str) -> list[dict]:
    documents = []

    for path in Path(directory).rglob("*"):
        if path.suffix not in {".txt", ".md"}:
            continue

        documents.append(
            {
                "id": path.stem,
                "text": path.read_text(encoding="utf-8"),
                "source": str(path),
            }
        )

    return documents