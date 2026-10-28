# RAG hỏi-đáp. Chạy:
#   python -m hybrid_graphrag.main novel --source Novel-25646 --type "Complex Reasoning" "How does alcohol affect the brain?"
#   python -m hybrid_graphrag.main medical            (không có câu hỏi -> chế độ hỏi liên tục)
import argparse
import sys

from hybrid_graphrag.config import LLM_MODEL, RETRIEVAL_BY_TYPE
from hybrid_graphrag.generation.llm_client import call_gemini
from hybrid_graphrag.prompts.rag import build_prompt
from hybrid_graphrag.retrieval.vector_search import get_db, vector_search

DEFAULT_CFG = {"top_k": 10, "max_output_tokens": 512}


def retrieve_vector(question: str, domain: str, source: str | None, top_k: int) -> dict:
    return {"chunks": vector_search(get_db(domain, source), question, top_k), "facts": []}


# Thêm cách truy xuất mới vào đây, ví dụ "hybrid": retrieve_hybrid (dual_search) khi làm xong.
RETRIEVERS = {"vector": retrieve_vector}


def answer_question(
    question: str,
    domain: str,
    source: str | None = None,
    question_type: str | None = None,
    top_k: int | None = None,
    model: str = LLM_MODEL,
    mode: str = "vector",
) -> dict:
    if mode not in RETRIEVERS:
        raise ValueError(f"mode '{mode}' chưa có; các mode hiện có: {list(RETRIEVERS)}")
    cfg = RETRIEVAL_BY_TYPE.get(question_type, DEFAULT_CFG)
    k = top_k or cfg["top_k"]
    ret = RETRIEVERS[mode](question, domain, source, k)
    system, user = build_prompt(question, ret["chunks"], question_type, facts=ret.get("facts") or None)
    answer = call_gemini(user, system, model=model, max_output_tokens=cfg["max_output_tokens"])
    return {"answer": answer, "chunks": ret["chunks"], "facts": ret.get("facts", []), "top_k": k, "mode": mode}


def cli() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["medical", "novel"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--source", help="bắt buộc với novel, ví dụ Novel-25646")
    ap.add_argument("--type", dest="qtype", choices=list(RETRIEVAL_BY_TYPE))
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--mode", default="vector", choices=list(RETRIEVERS))
    a = ap.parse_args()
    if a.domain == "novel" and not a.source:
        ap.error("novel cần --source")
    source = a.source if a.domain == "novel" else None

    def ask(q: str) -> None:
        r = answer_question(q, a.domain, source, a.qtype, a.top_k, mode=a.mode)
        print(f"\nAnswer: {r['answer']}\n\nTop chunks (k={r['top_k']}, mode={r['mode']}):")
        for c in r["chunks"][:5]:
            score = c.get("vector_score", c.get("score", 0.0))
            print(f"  [{score:.3f}] {c['chunk_id']}: {c['text'][:100]}...")

    if a.question:
        ask(a.question)
    else:
        while (q := input("\nCâu hỏi (Enter để thoát): ").strip()):
            ask(q)


if __name__ == "__main__":
    cli()