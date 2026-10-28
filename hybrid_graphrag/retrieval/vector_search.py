from functools import lru_cache

from hybrid_graphrag.config import index_dir
from hybrid_graphrag.knowledge_base.embeddings import embed_query
from hybrid_graphrag.knowledge_base.vector_db import VectorDB


@lru_cache(maxsize=8)
def get_db(domain: str, source: str | None = None) -> VectorDB:
    """medical -> artifacts/medical ; novel -> artifacts/novel/<source>."""
    if domain == "novel" and not source:
        raise ValueError("domain 'novel' cần source, ví dụ Novel-25646")
    path = index_dir(domain, source if domain == "novel" else None)
    if not VectorDB.exists(path):
        hint = f"--source {source}" if domain == "novel" else ""
        raise FileNotFoundError(
            f"Chưa có index tại {path}. Build: python -m hybrid_graphrag.knowledge_base.build_knowledge_base {domain} {hint}"
        )
    return VectorDB.load(path)


def vector_search(db: VectorDB, query: str, top_k: int) -> list[dict]:
    """[{**chunk, 'vector_score'}, ...] giảm dần."""
    return db.search(embed_query(query), top_k)