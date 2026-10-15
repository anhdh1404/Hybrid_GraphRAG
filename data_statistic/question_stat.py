"""
Thống kê bộ câu hỏi GraphRAG-Bench theo từng loại câu hỏi.
Chạy: python -m data_statistic.question_stat
"""
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "GraphRAG-Bench" / "Datasets" / "Questions"
OUT = Path(__file__).with_name("question_stats.json")
FILES = ["medical_questions.json", "novel_questions.json"]

SENT = re.compile(r"(?<=[.!?])\s+")


def n_sent(text: str) -> int:
    return len([s for s in SENT.split(text.strip()) if s.strip()]) if text else 0


def n_evidence(ev) -> int:
    if not ev:
        return 0
    return len(ev) if isinstance(ev, list) else 1


def describe(v: list[float]) -> dict:
    if not v:
        return {}
    p90 = statistics.quantiles(v, n=10)[-1] if len(v) > 1 else v[0]
    return {
        "mean": round(statistics.mean(v), 1),
        "median": statistics.median(v),
        "p90": round(p90, 1),
        "min": min(v),
        "max": max(v),
    }


def analyze(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        items = [x for x in json.load(f) if isinstance(x, dict)]

    groups = defaultdict(lambda: defaultdict(list))
    examples = {}
    sources = Counter()
    for it in items:
        t = str(it.get("question_type") or "unknown")
        q, a = it.get("question", ""), it.get("answer", "")
        g = groups[t]
        g["q_words"].append(len(q.split()))
        g["a_words"].append(len(a.split()))
        g["a_sents"].append(n_sent(a))
        g["evidence"].append(n_evidence(it.get("evidence")))
        if it.get("source"):
            sources[str(it["source"])] += 1
        # lưu 1 ví dụ có answer dài nhất của mỗi loại
        if t not in examples or len(a) > len(examples[t]["answer"]):
            examples[t] = {"id": it.get("id"), "question": q, "answer": a}

    return {
        "file": path.name,
        "num_questions": len(items),
        "fields": sorted(items[0].keys()) if items else None,
        "by_type": {
            t: {"count": len(g["q_words"]), **{k: describe(v) for k, v in g.items()}}
            for t, g in groups.items()
        },
        "sources": dict(sources.most_common()),
        "longest_answer_example": examples,
    }


def print_report(r: dict):
    print(f"\n{'=' * 78}\n{r['file']}  |  {r['num_questions']:,} câu  |  fields: {r['fields']}\n{'=' * 78}")
    print(f"{'type':<22}{'n':>6}  {'a_words (med/p90/max)':<24}{'a_sents (med/max)':<20}{'evidence (mean)':<10}")
    for t, s in sorted(r["by_type"].items(), key=lambda x: -x[1]["count"]):
        aw, asn, ev = s["a_words"], s["a_sents"], s["evidence"]
        print(f"{t:<22}{s['count']:>6}  "
              f"{aw['median']:>6}/{aw['p90']:>6}/{aw['max']:<8}"
              f"{asn['median']:>5}/{asn['max']:<12}"
              f"{ev['mean']:<10}")
    if r["sources"]:
        print(f"\nsource: {len(r['sources'])} giá trị; top 5: {list(r['sources'].items())[:5]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    reports = []
    for name in FILES:
        path = QDIR / name
        if not path.exists():
            print(f"Cảnh báo: không thấy {path}")
            continue
        r = analyze(path)
        reports.append(r)
        print_report(r)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu: {OUT}")


if __name__ == "__main__":
    main()