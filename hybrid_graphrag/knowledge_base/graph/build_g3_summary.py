# G3a: tóm tắt từng community (node H) bằng Qwen qua vLLM. Resume theo cụm.
# Đầu vào mỗi index: g2_communities_<tag>.json + g1_units_<tag>.jsonl (không cần .gml).
#   python -m hybrid_graphrag.knowledge_base.graph.build_g3_summary medical --workers 32
#   python -m hybrid_graphrag.knowledge_base.graph.build_g3_summary novel --workers 32
#   python -m hybrid_graphrag.knowledge_base.graph.build_g3_summary novel --source Novel-2544 --max-communities 10
import argparse
import glob
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hybrid_graphrag.config import VLLM_WORKERS, index_dir
from hybrid_graphrag.data.loader import load_corpus
from hybrid_graphrag.generation.llm_client import LLMError
from hybrid_graphrag.generation.vllm_client import (
    VLLMOutputError, call_vllm_text, get_vllm_usage, vllm_server_ready,
)
from hybrid_graphrag.prompts.summary import build_summary_prompt

MAX_UNITS, MAX_RELS = 30, 60
MAX_CONSECUTIVE_ERRORS = 20


def load_inputs(base: Path, suffix: str):
    cf = sorted(glob.glob(str(base / f"g2_communities_*-{suffix}.json")))
    uf = sorted(glob.glob(str(base / f"g1_units_*-{suffix}.jsonl")))
    if not cf or not uf:
        raise FileNotFoundError(f"Thiếu g2_communities_*-{suffix}.json hoặc g1_units_*-{suffix}.jsonl trong {base}")
    comms = json.loads(Path(cf[0]).read_text(encoding="utf-8"))["communities"]
    units = {}
    for line in Path(uf[0]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            for j, u in enumerate(rec["units"]):
                units[f"U:{rec['chunk_id']}#{j}"] = u
    tag = Path(cf[0]).stem.removeprefix("g2_communities_")
    return comms, units, tag


def build_prompt(c: dict, units: dict) -> tuple[str, str]:
    us = [units[u] for u in c["units"] if u in units]
    us.sort(key=lambda u: len(u["entities"]), reverse=True)  # cụm > 30 unit: lấy 30 unit nhiều entity nhất
    us = us[:MAX_UNITS]
    rels, seen = [], set()
    for u in us:
        for r in u["relations"]:
            t = (r["source"], r["relation"], r["target"])
            if t not in seen:
                seen.add(t)
                rels.append(t)
    subject = "Main entities: " + ", ".join(c["top_entities"][:6])
    return build_summary_prompt(subject, [u["summary"] for u in us], rels, MAX_UNITS, MAX_RELS)


def read_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(l)["community"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def summarize(system: str, user: str) -> str:
    return call_vllm_text(user, system)


def process(domain: str, source: str | None, a) -> None:
    base = index_dir(domain, source if domain == "novel" else None)
    comms, units, tag = load_inputs(base, a.suffix)
    out = base / f"g3_summaries_{tag}.jsonl"
    done = read_done(out)
    todo = [c for c in comms if c["community"] not in done]
    if a.max_communities:
        todo = todo[:a.max_communities]
    print(f"\n===== {source or domain}: {len(done)}/{len(comms)} cụm đã xong, xử lý {len(todo)} =====")
    if not todo:
        return

    skipped, consec, t0, words = 0, 0, time.time(), 0
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(summarize, *build_prompt(c, units)): c for c in todo}
        try:
            for n, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    text = fut.result()
                except VLLMOutputError as e:
                    skipped += 1
                    consec += 1
                    print(f"  [bỏ qua] cụm {c['community']}: {e}")
                    if consec >= MAX_CONSECUTIVE_ERRORS:
                        raise LLMError(f"{MAX_CONSECUTIVE_ERRORS} cụm lỗi liên tiếp, nghi lỗi cấu hình")
                    continue
                consec = 0
                words += len(text.split())
                f.write(json.dumps({"community": c["community"], "description": text}, ensure_ascii=False) + "\n")
                f.flush()
                if n % 20 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} ({(time.time() - t0) / 60:.1f} phút)")
        except LLMError as e:
            ex.shutdown(wait=False, cancel_futures=True)
            print(f"\n[g3a] DỪNG: {e}\n[g3a] Kết quả đã lưu, chạy lại cùng lệnh để tiếp tục.")
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            print("\n[g3a] Ngắt tay. Kết quả đã lưu, chạy lại để tiếp tục.")
    ok = len(todo) - skipped
    print(f"[g3a] xong {ok}/{len(todo)} cụm, bỏ qua {skipped}; TB {words / max(ok, 1):.0f} từ/tóm tắt")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["medical", "novel"])
    ap.add_argument("--source", help="novel: một cuốn (mặc định: cả 20)")
    ap.add_argument("--suffix", default="full")
    ap.add_argument("--workers", type=int, default=VLLM_WORKERS)
    ap.add_argument("--max-communities", type=int, help="chỉ xử lý N cụm chưa xong (để thử nhỏ)")
    a = ap.parse_args()
    if not vllm_server_ready():
        sys.exit("Không thấy server vLLM. Khởi động server trước.")
    targets = [None] if a.domain == "medical" else (
        [a.source] if a.source else [d["corpus_name"] for d in load_corpus("novel")])
    for s in targets:
        process(a.domain, s, a)
    u = get_vllm_usage()
    print(f"\n[g3a] token vLLM: {u['calls']} lượt, {u['prompt_tokens']:,} vào, {u['output_tokens']:,} ra")


if __name__ == "__main__":
    main()