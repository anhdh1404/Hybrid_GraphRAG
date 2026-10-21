import sys

from hybrid_graphrag.config import index_dir
from hybrid_graphrag.data.loader import load_questions
from hybrid_graphrag.knowledge_base.embeddings import embed_query
from hybrid_graphrag.knowledge_base.vector_db import VectorDB

sys.stdout.reconfigure(encoding="utf-8")
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "Novel-25646"

db = VectorDB.load(index_dir("novel", SOURCE))
print("meta:", db.meta)

for qtype in ("Fact Retrieval", "Complex Reasoning"):
    qs = load_questions("novel", source=SOURCE, question_type=qtype)
    if not qs:
        print(f"\n[{qtype}] không có câu hỏi")
        continue
    q = qs[0]
    qv = embed_query(q["question"])
    print(f"\n=== {qtype} ===\nQ: {q['question']}\nA: {q['answer']}\nEvidence: {q['evidence']}")
    hits = db.search(qv, 3)
    for r in hits:
        print(f"\n[{r['vector_score']:.3f}] {r['chunk_id']}\n{r['text'][:300]}")
    chk = db.score(qv, [h["chunk_id"] for h in hits])
    assert all(abs(chk[h["chunk_id"]] - h["vector_score"]) < 1e-4 for h in hits), "score() lệch search()"