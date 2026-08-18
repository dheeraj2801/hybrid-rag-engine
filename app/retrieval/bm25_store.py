import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    Simple baseline tokenizer.

    Lowercases text and extracts alphanumeric tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[dict]:

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:k]:
            chunk = self.chunks[index]

            results.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "score": float(scores[index]),
                    "metadata": {
                        "chunk_id": chunk["id"],
                        "source": chunk["source"],
                        "parent_id": chunk["parent_id"],
                    },
                }
            )

        return results