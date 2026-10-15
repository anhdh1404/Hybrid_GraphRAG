"""
Thống kê corpus GraphRAG-Bench (medical, novel) phục vụ lên kế hoạch pipeline.
Chạy từ thư mục gốc:  python -m data_statistic.dataset_stat
Kết quả in ra màn hình và lưu vào data_statistic/dataset_stats.json
"""
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "GraphRAG-Bench" / "Datasets" / "Corpus"
OUT_PATH = Path(__file__).with_name("dataset_stats.json")

CORPUS_FILES = ["medical.json", "novel.json"]

# TODO: sau này import từ hybrid_graphrag/config để dùng chung với preprocessor.py
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
CHARS_PER_TOKEN = 4          # ước lượng thô cho tiếng Anh
PROMPT_OVERHEAD_TOKENS = 800  # prompt trích entity/relation ở G1
N_SHORTEST = 3                # số cuốn ngắn nhất đánh dấu để chạy thử

TEXT_FIELDS = ["context", "text", "content", "body", "document", "passage", "abstract"]
NAME_FIELDS = ["corpus_name", "name", "title", "id"]
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ----------------------------------------------------------------- helpers
def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def first_field(d: dict, candidates: list[str]):
    for key in candidates:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return None


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "total": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def extract_documents(data) -> list:
    """Chuẩn hóa JSON về list document."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("documents", "data"):
            if key in data:
                return data[key]
        # dict đơn chứa trực tiếp trường văn bản -> 1 document
        if first_field(data, TEXT_FIELDS) is not None:
            return [data]
        return list(data.values())
    return []


def doc_text(doc) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        text = first_field(doc, TEXT_FIELDS)
        if text:
            return str(text)
        # fallback: nối các chuỗi, bỏ các trường tên/id
        return " ".join(
            v for k, v in doc.items() if isinstance(v, str) and k not in NAME_FIELDS
        )
    return ""


def doc_name(doc, i: int) -> str:
    if isinstance(doc, dict):
        name = first_field(doc, NAME_FIELDS)
        if name:
            return str(name)
    return f"doc{i}"


def est_chunks(n_chars: int) -> int:
    """Số chunk ước lượng, có tính overlap (khớp với preprocessor.py)."""
    if n_chars <= CHUNK_SIZE:
        return 1
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return math.ceil((n_chars - CHUNK_OVERLAP) / step)


# ------------------------------------------------------------ corpus stats
def corpus_stats(path: Path) -> dict:
    docs = extract_documents(load_json(path))
    pairs = [(d, doc_text(d)) for d in docs]
    pairs = [(d, t) for d, t in pairs if t]

    stats = {
        "file": path.name,
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "num_documents": len(docs),
        "document_fields": sorted(docs[0].keys()) if docs and isinstance(docs[0], dict) else None,
    }
    if not pairs:
        stats["error"] = "Không tìm thấy văn bản"
        return stats

    texts = [t for _, t in pairs]
    chars = [len(t) for t in texts]
    words = [len(t.split()) for t in texts]
    total_chars = sum(chars)
    total_sentences = sum(len([s for s in SENT_SPLIT.split(t) if s.strip()]) for t in texts)
    n_chunks = sum(est_chunks(c) for c in chars)
    est_tokens = total_chars // CHARS_PER_TOKEN

    stats["characters"] = describe(chars)
    stats["words"] = describe(words)
    stats["pipeline_estimates"] = {
        "num_sentences": total_sentences,
        "avg_sentence_chars": total_chars / max(total_sentences, 1),
        "est_tokens": est_tokens,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "est_chunks": n_chunks,
        # G1 gọi LLM 1 lần/chunk
        "g1_llm_calls": n_chunks,
        "g1_est_input_tokens": est_tokens + n_chunks * PROMPT_OVERHEAD_TOKENS,
    }

    if len(pairs) > 1:
        per_doc = [
            {
                "index": i,
                "name": doc_name(d, i),
                "characters": len(t),
                "words": len(t.split()),
                "est_chunks": est_chunks(len(t)),
            }
            for i, (d, t) in enumerate(pairs)
        ]
        shortest = {x["index"] for x in sorted(per_doc, key=lambda x: x["words"])[:N_SHORTEST]}
        for x in per_doc:
            x["test_subset"] = x["index"] in shortest
        stats["per_document"] = per_doc
    return stats


# ------------------------------------------------------------------ report
def fmt(x) -> str:
    return f"{x:,.1f}" if isinstance(x, float) else f"{x:,}"


def print_desc(label: str, d: dict):
    if not d:
        return
    print(
        f"  {label:<8} total={fmt(d['total'])}  mean={fmt(d['mean'])}  "
        f"median={fmt(d['median'])}  min={fmt(d['min'])}  max={fmt(d['max'])}  "
        f"std={fmt(d['std'])}"
    )


def print_report(report: dict):
    line = "=" * 78
    print(f"\n{line}\nCORPUS\n{line}")
    for s in report["corpus"]:
        print(f"\n[{s['file']}]  {s['file_size_mb']:.2f} MB  |  "
              f"{s['num_documents']} document  |  fields: {s['document_fields']}")
        if "error" in s:
            print("  Lỗi:", s["error"])
            continue
        if s["num_documents"] > 1:
            print_desc("chars", s["characters"])
            print_desc("words", s["words"])
        else:
            print(f"  chars total = {fmt(s['characters']['total'])}")
            print(f"  words total = {fmt(s['words']['total'])}")
        e = s["pipeline_estimates"]
        print(f"  {fmt(e['num_sentences'])} câu (TB {e['avg_sentence_chars']:.0f} ký tự/câu), "
              f"~{fmt(e['est_tokens'])} token")
        print(f"  chunk {e['chunk_size']}/overlap {e['chunk_overlap']}: ~{fmt(e['est_chunks'])} chunk")
        print(f"  G1: ~{fmt(e['g1_llm_calls'])} lần gọi LLM, "
              f"~{fmt(e['g1_est_input_tokens'])} token đầu vào (đã cộng overhead prompt)")
        if "per_document" in s:
            print("  từng document (sắp theo số từ; * = subset chạy thử):")
            for d in sorted(s["per_document"], key=lambda x: x["words"]):
                mark = "*" if d["test_subset"] else " "
                print(f"   {mark} #{d['index']:<3} {d['name'][:30]:<30} "
                      f"{fmt(d['words']):>9} từ  {fmt(d['est_chunks']):>6} chunk")

    c = report["combined"]
    print(f"\n{line}\nTỔNG\n{line}")
    print(f"  Corpus: {c['corpus_files']} file | {c['total_documents']:,} document | "
          f"{c['total_size_mb']:.2f} MB | ~{c['total_chunks']:,} chunk\n")


# -------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    corpus = []
    for name in CORPUS_FILES:
        path = CORPUS_DIR / name
        if not path.exists():
            print(f"Cảnh báo: không thấy {path}")
            continue
        print(f"Đang phân tích {name} ...")
        corpus.append(corpus_stats(path))

    report = {
        "corpus": corpus,
        "combined": {
            "corpus_files": len(corpus),
            "total_documents": sum(s["num_documents"] for s in corpus),
            "total_size_mb": sum(s["file_size_mb"] for s in corpus),
            "total_chunks": sum(s.get("pipeline_estimates", {}).get("est_chunks", 0) for s in corpus),
        },
    }
    print_report(report)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu: {OUT_PATH}")


if __name__ == "__main__":
    main()