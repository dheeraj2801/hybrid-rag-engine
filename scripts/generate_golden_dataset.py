import json
from pathlib import Path

OUT = Path("evaluation/datasets/golden_dataset.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

categories = ["semantic", "exact", "technical", "error_code", "multi_concept"]

# pools of chunk ids to map to
kafka_chunks = ["kafka_0", "kafka_1", "kafka_2", "kafka_3"]
fastapi_chunks = ["fastapi_0", "fastapi_1", "fastapi_2"]
python_chunks = ["python_0", "python_1", "python_2", "python_3"]

entries = []

def pick_for_category(cat, idx):
    if cat == "semantic":
        pool = [kafka_chunks[idx % len(kafka_chunks)], fastapi_chunks[idx % len(fastapi_chunks)], python_chunks[idx % len(python_chunks)]]
        return [pool[idx % len(pool)]]
    if cat == "exact":
        pool = [fastapi_chunks[0], kafka_chunks[0], python_chunks[0]]
        return [pool[idx % len(pool)]]
    if cat == "technical":
        pool = [kafka_chunks[0], kafka_chunks[1], python_chunks[1], fastapi_chunks[0]]
        return [pool[idx % len(pool)]]
    if cat == "error_code":
        pool = [kafka_chunks[2], fastapi_chunks[0], python_chunks[0]]
        return [pool[idx % len(pool)]]
    if cat == "multi_concept":
        # return two-chunk list
        a = kafka_chunks[idx % len(kafka_chunks)]
        b = python_chunks[idx % len(python_chunks)]
        return [a, b]

for i in range(150):
    qid = f"q{(i+1):03d}"
    cat = categories[i // 30]
    num = i + 1
    query_text = f"[{cat}] example query number {num} about {cat} topics"
    relevant = pick_for_category(cat, i)
    ref = f"Reference answer for {qid} in category {cat}."
    entry = {"id": qid, "category": cat, "query": query_text, "relevant_chunk_ids": relevant, "reference_answer": ref}
    entries.append(entry)

with OUT.open("w", encoding="utf-8") as fh:
    for e in entries:
        fh.write(json.dumps(e) + "\n")

print(f"Wrote {len(entries)} queries to {OUT}")
