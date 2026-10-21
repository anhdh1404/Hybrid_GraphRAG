# Entry point xây Knowledge Base. Chạy ở thư mục gốc repo:
#   python -m hybrid_graphrag.knowledge_base.build_knowledge_base medical
#   python -m hybrid_graphrag.knowledge_base.build_knowledge_base novel --source Novel-25646
import argparse
import sys

from hybrid_graphrag.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL, index_dir,
)
from hybrid_graphrag.data.loader import load_corpus
from hybrid_graphrag.data.preprocessor import make_chunks
from hybrid_graphrag.knowledge_base.embeddings import embed_texts
from hybrid_graphrag.knowledge_base.vector_db import VectorDB


def _params() -> dict:
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


# 1. Vector database: medical -> artifacts/medical ; novel -> artifacts/novel/<corpus_name>
def build_vector_db(domain: str, source: str | None = None, force: bool = False) -> None:
    params = _params()
    for doc in load_corpus(domain, source):
        name = doc["corpus_name"]
        out = index_dir(domain, name if domain == "novel" else None)

        if VectorDB.exists(out) and not force:
            old = VectorDB.read_meta(out)
            if all(old.get(k) == v for k, v in params.items()):
                print(f"[skip] {name}: đã có index cùng tham số")
                continue
            print(f"[warn] {name}: tham số đổi, build lại")

        chunks = make_chunks(doc)
        embeddings = embed_texts([c["text"] for c in chunks])
        db = VectorDB.build(embeddings, chunks, {**params, "corpus_name": name})
        db.save(out)
        print(f"[ok] {name}: {db.index.ntotal} chunk, dim={db.dim} -> {out}")


# 2. Knowledge graph (G1 -> G2 -> G3)
# TODO: build_graph(domain, source, force), gọi sau khi nhánh vector đã ổn


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["medical", "novel"])
    ap.add_argument("--source", help="chỉ build một cuốn novel, ví dụ Novel-25646")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build_vector_db(a.domain, a.source, a.force)