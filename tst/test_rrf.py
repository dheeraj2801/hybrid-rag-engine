from app.retrieval.vector_service import search as vsearch
from app.retrieval.bm25_service import bm25_retriever
from app.retrieval.rrf import rrf_fuse


def main():
    q = "How does Kafka consumer rebalancing work?"

    v = vsearch(q, k=10)
    b = bm25_retriever.search(q, k=10)

    fused = rrf_fuse([v, b], k=60, top_n=10)

    print('\nVector top-5:')
    for i, r in enumerate(v[:5], start=1):
        print(f"{i}. {r['id']}")

    print('\nBM25 top-5:')
    for i, r in enumerate(b[:5], start=1):
        print(f"{i}. {r['id']}")

    print('\nRRF fused top-10:')
    for i, r in enumerate(fused, start=1):
        print(f"{i}. {r['id']} (score={r['score']:.6f})")


if __name__ == '__main__':
    main()
