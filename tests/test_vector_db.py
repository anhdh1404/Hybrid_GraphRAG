import numpy as np
import pytest

from hybrid_graphrag.knowledge_base.vector_db import VectorDB


def make_db(n: int = 20, dim: int = 8) -> VectorDB:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, dim)).astype("float32")
    chunks = [
        {"chunk_id": f"T_{i:05d}", "source": "T", "index": i, "text": f"chunk {i}"}
        for i in range(n)
    ]
    return VectorDB.build(emb, chunks, {"corpus_name": "T"})


def test_search_returns_itself_first():
    db = make_db()
    q = db.get_embeddings(["T_00005"])[0]
    hits = db.search(q, 3)
    assert hits[0]["chunk_id"] == "T_00005"
    assert hits[0]["vector_score"] == pytest.approx(1.0, abs=1e-4)
    assert hits[0]["vector_score"] >= hits[1]["vector_score"] >= hits[2]["vector_score"]


def test_score_matches_search():
    db = make_db()
    q = db.get_embeddings(["T_00003"])[0]
    hits = db.search(q, 5)
    scores = db.score(q, [h["chunk_id"] for h in hits])
    for h in hits:
        assert scores[h["chunk_id"]] == pytest.approx(h["vector_score"], abs=1e-4)


def test_top_k_larger_than_index():
    db = make_db(n=4)
    q = db.get_embeddings(["T_00000"])[0]
    assert len(db.search(q, 100)) == 4


def test_save_load_roundtrip(tmp_path):
    db = make_db()
    db.save(tmp_path)
    assert VectorDB.exists(tmp_path)
    db2 = VectorDB.load(tmp_path)
    q = db.get_embeddings(["T_00007"])[0]
    assert [h["chunk_id"] for h in db.search(q, 5)] == [h["chunk_id"] for h in db2.search(q, 5)]
    assert db2.meta["n_chunks"] == 20


def test_build_mismatch_raises():
    emb = np.zeros((3, 8), dtype="float32")
    with pytest.raises(ValueError):
        VectorDB.build(emb, [{"chunk_id": "a"}])


def test_unknown_chunk_id_raises():
    db = make_db()
    with pytest.raises(KeyError):
        db.get_embeddings(["không_tồn_tại"])