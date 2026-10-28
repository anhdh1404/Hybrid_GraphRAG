# Chấm điểm câu trả lời của run_baseline (tách khỏi bước sinh nên chấm lại không tốn lượt sinh).
#   python -m experiments.run_evaluation artifacts/runs/baseline_vector_medical.jsonl
#   python -m experiments.run_evaluation <file1.jsonl> <file2.jsonl> --limit 20 --no-sim
import argparse
import json
import sys
from pathlib import Path

from experiments.metrics_gemini import METRICS_BY_TYPE, evaluate_sample, mean, std
from hybrid_graphrag.config import LLM_MODEL
from hybrid_graphrag.generation.llm_client import LLMBlockedError, LLMError

TYPE_ORDER = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]
METRIC_ORDER = ["answer_correctness", "rouge_l", "coverage", "faithfulness"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def contexts_of(rec: dict) -> list[str]:
    """Văn bản các chunk đã truy xuất, cộng thêm quan hệ từ đồ thị nếu có (chỉ faithfulness cần)."""
    from hybrid_graphrag.retrieval.vector_search import get_db

    domain = rec.get("domain") or ("novel" if str(rec["source"]).startswith("Novel") else "medical")
    db = get_db(domain, rec["source"] if domain == "novel" else None)
    return [db.get_chunk(cid)["text"] for cid in rec["retrieved"]] + list(rec.get("facts") or [])


def is_blocked(r: dict) -> bool:
    return bool(r.get("gen_blocked") or r.get("judge_blocked"))


def summarize(rows: list[dict]) -> dict:
    out = {}
    for t in TYPE_ORDER:
        rs = [r for r in rows if r["question_type"] == t]
        if not rs:
            continue
        scored = [r for r in rs if not is_blocked(r)]
        d = {"n": len(rs), "blocked": len(rs) - len(scored)}
        for m in METRICS_BY_TYPE[t]:
            vals = [r.get(m) for r in scored]
            d[m], d[m + "_std"] = mean(vals), std(vals)
        out[t] = d
    if all(t in out for t in TYPE_ORDER):  # quy ước leaderboard: ACC = trung bình không trọng số 4 loại
        out["ACC_avg"] = mean(out[t]["answer_correctness"] for t in TYPE_ORDER)
    return out


def print_summary(name: str, s: dict) -> None:
    print(f"\n{name}")
    print(f"{'type':<22}{'n':>5}{'blocked':>9}" + "".join(f"{m:>20}" for m in METRIC_ORDER))
    for t in TYPE_ORDER:
        if t in s:
            cells = "".join(
                f"{s[t][m]:>20.3f}" if s[t].get(m) is not None else f"{'-':>20}" for m in METRIC_ORDER
            )
            print(f"{t:<22}{s[t]['n']:>5}{s[t]['blocked']:>9}{cells}")
    if s.get("ACC_avg") is not None:
        print(f"ACC trung bình 4 loại (answer_correctness): {s['ACC_avg']:.3f}")


def evaluate_file(path: Path, model: str, use_sim: bool, limit: int | None) -> None:
    out_path = path.with_name(path.stem + "_eval.jsonl")
    recs = read_jsonl(path)
    cfg = f"{model}|sim={use_sim}"
    done = {r["id"]: r for r in read_jsonl(out_path)}  # nếu trùng id, bản ghi sau thắng
    todo = [r for r in recs if r["id"] not in done or done[r["id"]].get("cfg") != cfg]
    if limit:
        todo = todo[:limit]
    print(f"[{path.name}] {len(recs)} câu, cần chấm {len(todo)} (judge={model}, sim={use_sim})")

    try:
        with open(out_path, "a", encoding="utf-8") as f:
            for i, rec in enumerate(todo, 1):
                row = {"id": rec["id"], "question_type": rec["question_type"], "cfg": cfg}
                if rec.get("blocked") or not str(rec.get("answer", "")).strip():
                    row["gen_blocked"] = True  # câu trả lời bị chặn hoặc rỗng: không chấm
                else:
                    try:
                        row.update(evaluate_sample(rec, model, use_sim, get_contexts=lambda r=rec: contexts_of(r)))
                    except LLMBlockedError:
                        row["judge_blocked"] = True  # judge bị chặn ở câu này: bỏ qua, chạy tiếp
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done[rec["id"]] = row
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")
    except LLMError as e:
        print(f"\n[eval] DỪNG: {e}\n[eval] Kết quả đã lưu, chạy lại cùng lệnh để tiếp tục.")
    except KeyboardInterrupt:
        print("\n[eval] Ngắt tay. Kết quả đã lưu, chạy lại để tiếp tục.")

    rows = [done[r["id"]] for r in recs if r["id"] in done and done[r["id"]].get("cfg") == cfg]
    summary = summarize(rows)
    print_summary(f"{path.stem}  ({len(rows)}/{len(recs)} câu đã chấm)", summary)
    summary_path = path.with_name(path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu: {summary_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="file .jsonl do run_baseline tạo ra")
    ap.add_argument("--judge-model", default=LLM_MODEL, help="giữ cố định giữa các hệ thống để so sánh công bằng")
    ap.add_argument("--no-sim", action="store_true", help="answer correctness chỉ dùng F-beta, bỏ phần tương đồng embedding")
    ap.add_argument("--limit", type=int, help="chỉ chấm N câu chưa xong (để thử nhỏ)")
    a = ap.parse_args()
    for f in a.files:
        evaluate_file(Path(f), a.judge_model, not a.no_sim, a.limit)


if __name__ == "__main__":
    main()