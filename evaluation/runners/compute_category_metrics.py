import json
from pathlib import Path

DS = Path("evaluation/datasets/golden_dataset.jsonl")
RESULTS = Path("evaluation/results")

# load id -> category
id_to_cat = {}
with DS.open("r", encoding="utf-8") as fh:
    for line in fh:
        obj = json.loads(line)
        id_to_cat[obj["id"]] = obj.get("category", "unknown")

modes = ["vector.json", "bm25.json", "rrf.json"]
summary = {}
for mode_file in modes:
    path = RESULTS / mode_file
    data = json.loads(path.read_text(encoding="utf-8"))
    per = data.get("per_query", [])
    # aggregate per category
    cats = {}
    for q in per:
        qid = q["id"]
        cat = id_to_cat.get(qid, "unknown")
        cats.setdefault(cat, []).append(q.get("recall@5", 0.0))
    cat_summary = {cat: (sum(vals) / len(vals) if vals else 0.0) for cat, vals in cats.items()}
    summary[mode_file.replace('.json','')] = cat_summary

# write summary
out = RESULTS / "category_summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

# print markdown table
cats_order = sorted({c for m in summary.values() for c in m.keys()})
print("| Category | Vector@5 | BM25@5 | RRF@5 |")
print("|---|---:|---:|---:|")
for cat in cats_order:
    v = summary["vector"].get(cat, 0.0)
    b = summary["bm25"].get(cat, 0.0)
    r = summary["rrf"].get(cat, 0.0)
    print(f"| {cat} | {v:.3f} | {b:.3f} | {r:.3f} |")

print(f"Wrote category summary to {out}")
