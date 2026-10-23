# Đo evidence recall@k của baseline vector bằng "chunk vàng" (độ trùng từ nội dung).
#   python -m experiments.vector_recall                       (mặc định: Novel-25646)
#   python -m experiments.vector_recall Novel-25646 Novel-4128
#   python -m experiments.vector_recall novel-all             (cả 20 cuốn novel)
#   python -m experiments.vector_recall medical
import re
import sys
from collections import defaultdict

from hybrid_graphrag.config import index_dir
from hybrid_graphrag.data.loader import load_corpus, load_questions
from hybrid_graphrag.knowledge_base.embeddings import embed_queries
from hybrid_graphrag.knowledge_base.vector_db import VectorDB

KS = (5, 10, 20)
GROUNDED_MIN = 0.6  # evidence có chunk trùng >= 60% từ nội dung mới được coi là đánh giá được
STOP = set(
    "the and for are was were been being has had have does did will would can could should may "
    "might which who whom what when where why how than then that this these those with from into "
    "about over also there their they them his her its you your our not but all any each such "
    "more most some very just only own same too".split()
)


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}


def evidence_list(q: dict) -> list[str]:
    ev = q.get("evidence") or []
    return [ev] if isinstance(ev, str) else [e for e in ev if isinstance(e, str) and e.strip()]


def new_row() -> dict:
    return {"ev": 0, "grounded": 0, "best_sum": 0.0, **{k: 0 for k in KS}, "examples": []}


def run(domain: str, sources: list) -> None:
    """domain 'medical': sources = [None]; domain 'novel': sources = danh sách corpus_name."""
    rows = defaultdict(new_row)
    used, missing = [], []
    for src in sources:
        path = index_dir(domain, src if domain == "novel" else None)
        label = src or domain
        if not VectorDB.exists(path):
            missing.append(label)
            continue
        used.append(label)
        db = VectorDB.load(path)
        chunk_toks = [toks(c["text"]) for c in db.chunks]
        qs = [q for q in load_questions(domain, source=src) if evidence_list(q)]
        print(f"[{label}] {len(db.chunks)} chunk, {len(qs)} câu hỏi")
        qvecs = embed_queries([q["question"] for q in qs], show_progress=True)
        for q, qv in zip(qs, qvecs):
            top_rows = [db.id_to_row[h["chunk_id"]] for h in db.search(qv, max(KS))]
            r = rows[q["question_type"]]
            for ev in evidence_list(q):
                t = toks(ev)
                if not t:
                    continue
                overlaps = [len(t & ct) / len(t) for ct in chunk_toks]
                best = max(overlaps)
                r["ev"] += 1
                r["best_sum"] += best
                if best < GROUNDED_MIN:
                    if len(r["examples"]) < 2:
                        r["examples"].append((best, ev))
                    continue
                r["grounded"] += 1
                gold = {i for i, o in enumerate(overlaps) if o >= best - 1e-9}
                for k in KS:
                    r[k] += bool(gold & set(top_rows[:k]))

    if missing:
        print(f"\n[cảnh báo] thiếu index, đã bỏ qua: {', '.join(missing)}")
    if not rows:
        print("Không có dữ liệu để đo.")
        return

    print(f"\n{domain}: {len(used)} index (tính trên từng câu evidence; R@k chỉ tính evidence grounded)")
    head = f"{'type':<22}{'evid':>6}{'grounded':>10}{'mean_ov':>9}" + "".join(f"{'R@' + str(k):>8}" for k in KS)
    print(head)

    def line(name: str, r: dict) -> str:
        g = max(r["grounded"], 1)
        return (f"{name:<22}{r['ev']:>6}{r['grounded'] / r['ev']:>10.2f}{r['best_sum'] / r['ev']:>9.2f}"
                + "".join(f"{r[k] / g:>8.2f}" for k in KS))

    total = new_row()
    for t, r in sorted(rows.items(), key=lambda x: -x[1]["ev"]):
        print(line(t, r))
        for key in ("ev", "grounded", "best_sum", *KS):
            total[key] += r[key]
    print("-" * len(head))
    print(line("ALL", total))

    print("\nVí dụ evidence KHÔNG grounded (độ trùng thấp):")
    for t, r in rows.items():
        for best, ev in r["examples"]:
            print(f"  [{t}] ov={best:.2f}: {ev[:120]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args == ["medical"]:
        run("medical", [None])
    elif args == ["novel-all"]:
        run("novel", [d["corpus_name"] for d in load_corpus("novel")])
    else:
        run("novel", args or ["Novel-25646"])