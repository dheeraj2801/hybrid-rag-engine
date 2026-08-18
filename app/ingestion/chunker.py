from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []

    for document in documents:
        texts = splitter.split_text(document["text"])

        for index, text in enumerate(texts):
            chunks.append(
                {
                    "id": f"{document['id']}_{index}",
                    "text": text,
                    "source": document["source"],
                    "parent_id": document["id"],
                }
            )

    return chunks