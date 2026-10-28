# Sinh câu trả lời cho mẫu phân tầng; ghi JSONL, chạy lại sẽ bỏ qua câu đã làm.
#   python -m experiments.run_baseline medical --n-per-type 3
#   python -m experiments.run_baseline novel --n-per-type 3                 (lấy mẫu trên cả 20 cuốn)
#   python -m experiments.run_baseline novel --sources Novel-25646 Novel-4128 --n-per-type 3
#   --n-per-type 0 = toàn bộ câu hỏi; --top-k 20 để thử top_k khác (tên file tự thêm hậu tố _k20)
import argparse
import json
import random
import sys
from collections import defaultdict

from hybrid_graphrag.config import ARTIFACTS_DIR
from hybrid_graphrag.data.loader import load_questions
from hybrid_graphrag.generation.llm_client import LLMBlockedError, LLMError
from hybrid_graphrag.main import RETRIEVERS, answer_question


def stratified(questions: list[dict], n: int, seed: int = 42) -> list[dict]:
    """n câu mỗi loại; cố định theo seed + id nên các hệ thống khác nhau được chấm trên cùng tập câu."""
    if n <= 0:
        return sorted(questions, key=lambda q: q["id"])
    by_type = defaultdict(list)
    for q in sorted(questions, key=lambda q: q["id"]):
        by_type[q["question_type"]].append(q)
    rng = random.Random(seed)
    out = []
    for t in sorted(by_type):
        out += rng.sample(by_type[t], min(n, len(by_type[t])))
    return out


def read_done(path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["medical", "novel"])
    ap.add_argument("--sources", nargs="*", help="novel: chỉ lấy các cuốn này (mặc định: cả 20)")
    ap.add_argument("--n-per-type", type=int, default=20)
    ap.add_argument("--mode", default="vector", choices=list(RETRIEVERS))
    ap.add_argument("--top-k", type=int, help="ghi đè top_k theo loại câu hỏi")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", help="tên lượt chạy (mặc định: baseline_<mode>[_k<top_k>])")
    a = ap.parse_args()

    tag = a.tag or f"baseline_{a.mode}" + (f"_k{a.top_k}" if a.top_k else "")
    out_path = ARTIFACTS_DIR / "runs" / f"{tag}_{a.domain}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qs = load_questions(a.domain)
    if a.sources:
        qs = [q for q in qs if q["source"] in set(a.sources)]
    sample = stratified(qs, a.n_per_type, a.seed)
    done = read_done(out_path)
    todo = [q for q in sample if q["id"] not in done]
    print(f"[{tag}] mẫu {len(sample)} câu, đã xong {len(sample) - len(todo)}, còn {len(todo)}")

    try:
        with open(out_path, "a", encoding="utf-8") as f:
            for i, q in enumerate(todo, 1):
                src = q["source"] if a.domain == "novel" else None
                base = {
                    "id": q["id"], "domain": a.domain, "source": q["source"],
                    "question_type": q["question_type"], "question": q["question"],
                    "gold_answer": q["answer"], "mode": a.mode,
                }
                try:
                    r = answer_question(q["question"], a.domain, src, q["question_type"], a.top_k, mode=a.mode)
                    rec = {**base, "answer": r["answer"], "retrieved": [c["chunk_id"] for c in r["chunks"]],
                           "facts": r["facts"], "top_k": r["top_k"]}
                except LLMBlockedError:
                    rec = {**base, "answer": "", "blocked": True, "retrieved": [], "facts": [], "top_k": a.top_k}
                    print(f"  [bị chặn] {q['id']}")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {i}/{len(todo)} {q['question_type']}")
    except LLMError as e:
        print(f"\n[run] DỪNG: {e}\n[run] Kết quả đã lưu, chạy lại cùng lệnh để tiếp tục.")
    except KeyboardInterrupt:
        print("\n[run] Ngắt tay. Kết quả đã lưu, chạy lại để tiếp tục.")
    print(f"Đã lưu: {out_path}")


if __name__ == "__main__":
    main()