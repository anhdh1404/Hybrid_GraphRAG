# G1: trích semantic unit + entity + quan hệ từ từng chunk, rồi dựng đồ thị.
#   python -m hybrid_graphrag.knowledge_base.graph.build_g1 novel --source Novel-25646 --extractor vllm
#   python -m hybrid_graphrag.knowledge_base.graph.build_g1 novel --extractor vllm          (cả 20 cuốn)
#   python -m hybrid_graphrag.knowledge_base.graph.build_g1 medical --extractor vllm
#   --extractor gemini (mặc định): dùng Gemini, chạy tuần tự (tốn quota)
#   --max-chunks N: chỉ xử lý N chunk chưa xong của mỗi cuốn (để thử nhỏ)
#   --graph-only: không gọi LLM, chỉ dựng lại đồ thị từ kết quả đã có (nhớ truyền cùng --extractor/--model)
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx

from hybrid_graphrag.config import G1_MAX_TOKENS, LLM_MODEL, VLLM_MODEL, VLLM_WORKERS, index_dir
from hybrid_graphrag.data.loader import load_corpus
from hybrid_graphrag.data.preprocessor import make_chunks
from hybrid_graphrag.generation.llm_client import LLMBlockedError, LLMError, call_gemini_json
from hybrid_graphrag.generation.vllm_client import (
    VLLMOutputError, call_vllm_json, get_vllm_usage, vllm_server_ready,
)
from hybrid_graphrag.prompts.extraction import Extraction, build_extraction_prompt, clean_extraction

MAX_CONSECUTIVE_ERRORS = 20  # quá ngưỡng này thường là lỗi cấu hình (schema, prompt), không phải lỗi từng chunk


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")


def canon(entity: str) -> str:
    """Chuẩn hóa tên để gộp entity giữa các chunk: bỏ mạo từ đầu."""
    return re.sub(r"^(THE|A|AN) ", "", entity).strip()


def load_chunks(domain: str, source: str | None) -> list[dict]:
    """Chunk lấy thẳng từ corpus bằng đúng hàm chunking của index nên chunk_id luôn khớp."""
    docs = load_corpus(domain, source if domain == "novel" else None)
    return [c for d in docs for c in make_chunks(d)]


def read_done(path) -> dict[str, list[dict]]:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["chunk_id"]] = rec["units"]
    return done


def extract_chunk(extractor: str, model: str, text: str) -> dict:
    system, user = build_extraction_prompt(text)
    if extractor == "vllm":
        return call_vllm_json(user, system, Extraction, max_output_tokens=G1_MAX_TOKENS, model=model)
    return call_gemini_json(user, system, Extraction, model=model, max_output_tokens=G1_MAX_TOKENS)


def extract_all(units_path, chunks, extractor, model, workers, max_chunks) -> dict[str, list[dict]]:
    done = read_done(units_path)
    todo = [c for c in chunks if c["chunk_id"] not in done]
    if max_chunks is not None:
        todo = todo[:max_chunks]
    print(f"[g1] {len(done)}/{len(chunks)} chunk đã xong, sẽ xử lý {len(todo)} chunk "
          f"(extractor={extractor}, model={model}, workers={workers})")
    if not todo:
        return done

    stats: dict = {}
    skipped, consec, t0 = 0, 0, time.time()
    with open(units_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_chunk, extractor, model, c["text"]): c for c in todo}
        try:
            for n, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    data = fut.result()
                except (VLLMOutputError, LLMBlockedError) as e:
                    skipped += 1
                    consec += 1
                    print(f"  [bỏ qua] {c['chunk_id']}: {e}")
                    if consec >= MAX_CONSECUTIVE_ERRORS:
                        raise LLMError(f"{MAX_CONSECUTIVE_ERRORS} chunk lỗi liên tiếp, nghi lỗi cấu hình (schema/prompt)")
                    continue
                consec = 0
                units = clean_extraction(data, stats)
                f.write(json.dumps({"chunk_id": c["chunk_id"], "units": units}, ensure_ascii=False) + "\n")
                f.flush()
                done[c["chunk_id"]] = units
                if n % 10 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)}  ({n / el:.2f} chunk/giây, còn ~{(len(todo) - n) / (n / el) / 60:.0f} phút)")
        except LLMError as e:
            ex.shutdown(wait=False, cancel_futures=True)
            print(f"\n[g1] DỪNG: {e}\n[g1] Kết quả đã lưu, chạy lại cùng lệnh để tiếp tục.")
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            print("\n[g1] Ngắt tay. Kết quả đã lưu, chạy lại để tiếp tục.")
    print(f"[g1] bỏ qua {skipped} chunk lỗi (sẽ thử lại ở lần chạy sau); thống kê lọc: {stats}")
    return done


def build_graph(chunks: list[dict], extractions: dict[str, list[dict]]) -> nx.Graph:
    """Node: chunk (C:...), unit (U:...), entity (E:...).
    Cạnh: unit-chunk (from_chunk), unit-entity (mentions), entity-entity (relation, có weight)."""
    g = nx.Graph()
    for c in chunks:
        units = extractions.get(c["chunk_id"])
        if units is None:
            continue
        cid = f"C:{c['chunk_id']}"
        g.add_node(cid, kind="chunk", chunk_id=c["chunk_id"])
        for j, u in enumerate(units):
            uid = f"U:{c['chunk_id']}#{j}"
            g.add_node(uid, kind="unit", chunk_id=c["chunk_id"], text=u["summary"])
            g.add_edge(uid, cid, kind="from_chunk")
            for e in u["entities"]:
                eid = f"E:{canon(e)}"
                if eid not in g:
                    g.add_node(eid, kind="entity", name=canon(e))
                g.add_edge(uid, eid, kind="mentions")
            for r in u["relations"]:
                s, t = f"E:{canon(r['source'])}", f"E:{canon(r['target'])}"
                if s == t:
                    continue
                if g.has_edge(s, t) and g[s][t].get("kind") == "relation":
                    d = g[s][t]
                    d["weight"] += 1
                    if r["relation"] not in d["relation"].split(" | "):
                        d["relation"] += " | " + r["relation"]
                else:
                    g.add_edge(s, t, kind="relation", relation=r["relation"], weight=1)
    return g


def process(domain: str, source: str | None, a, model: str, workers: int) -> None:
    out = index_dir(domain, source if domain == "novel" else None)
    chunks = load_chunks(domain, source)
    tag = slug(model)
    units_path, graph_path = out / f"g1_units_{tag}.jsonl", out / f"g1_graph_{tag}.gml"
    print(f"\n===== {source or domain}: {len(chunks)} chunk -> {units_path.name} =====")

    extractions = read_done(units_path) if a.graph_only else extract_all(
        units_path, chunks, a.extractor, model, workers, a.max_chunks)
    g = build_graph(chunks, extractions)
    kinds = [d["kind"] for _, d in g.nodes(data=True)]
    print(f"[g1] đồ thị: {g.number_of_nodes()} node (chunk {kinds.count('chunk')}, unit {kinds.count('unit')}, "
          f"entity {kinds.count('entity')}), {g.number_of_edges()} cạnh; phủ {len(extractions)}/{len(chunks)} chunk")
    nx.write_gml(g, str(graph_path))
    print(f"[g1] đã lưu {graph_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["medical", "novel"])
    ap.add_argument("--source", help="novel: chỉ một cuốn (mặc định: cả 20 cuốn)")
    ap.add_argument("--extractor", choices=["gemini", "vllm"], default="gemini")
    ap.add_argument("--model", help="mặc định: VLLM_MODEL (vllm) hoặc LLM_MODEL (gemini)")
    ap.add_argument("--workers", type=int, default=VLLM_WORKERS, help="chỉ áp dụng cho vllm")
    ap.add_argument("--max-chunks", type=int)
    ap.add_argument("--graph-only", action="store_true")
    a = ap.parse_args()

    model = a.model or (VLLM_MODEL if a.extractor == "vllm" else LLM_MODEL)
    workers = a.workers if a.extractor == "vllm" else 1  # client Gemini không an toàn khi chạy đa luồng
    if a.extractor == "vllm" and not a.graph_only and not vllm_server_ready():
        sys.exit("Không thấy server vLLM. Khởi động server trước, hoặc dùng --graph-only.")

    targets = [None] if a.domain == "medical" else (
        [a.source] if a.source else [d["corpus_name"] for d in load_corpus("novel")])
    for src in targets:
        process(a.domain, src, a, model, workers)

    if a.extractor == "vllm" and not a.graph_only:
        u = get_vllm_usage()
        print(f"\n[g1] token vLLM: {u['calls']} lượt gọi, {u['prompt_tokens']:,} vào, {u['output_tokens']:,} ra")


if __name__ == "__main__":
    main()