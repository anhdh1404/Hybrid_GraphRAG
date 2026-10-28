import sys

from hybrid_graphrag.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL, index_dir,
)
from hybrid_graphrag.data.loader import load_corpus
from hybrid_graphrag.data.preprocessor import make_chunks
from hybrid_graphrag.knowledge_base.vector_db import VectorDB

sys.stdout.reconfigure(encoding="utf-8")
params = {
    "embedding_model": EMBEDDING_MODEL, "embedding_dim": EMBEDDING_DIM,
    "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
}
total, ok, bad = 0, 0, []
for domain in ("medical", "novel"):
    for doc in load_corpus(domain):
        name = doc["corpus_name"]
        out = index_dir(domain, name if domain == "novel" else None)
        if not VectorDB.exists(out):
            bad.append((name, "thiếu index"))
            continue
        meta = VectorDB.read_meta(out)
        n = len(make_chunks(doc))
        if meta.get("n_chunks") != n:
            bad.append((name, f"n_chunks {meta.get('n_chunks')} != {n}"))
            continue
        diff = [k for k, v in params.items() if meta.get(k) != v]
        if diff:
            bad.append((name, f"tham số lệch: {diff}"))
            continue
        ok += 1
        total += n
print(f"OK: {ok}/21 index, tổng {total:,} chunk (kỳ vọng 5,454)")
for name, why in bad:
    print("LỖI:", name, why)