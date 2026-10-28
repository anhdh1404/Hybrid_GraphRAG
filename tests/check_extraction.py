# Thử trích xuất G1 trên vài chunk: in kết quả, thống kê chất lượng và token thật.
#   python -m tests.check_extraction Novel-25646 21 64 76 --extractor vllm
#   python -m tests.check_extraction medical 5 200 600 --extractor vllm --quiet
#   python -m tests.check_extraction Novel-25646 21 64 76 --model gemini-3.1-flash-lite --no-cache
#   --price-in / --price-out (USD cho 1 triệu token): chỉ có ý nghĩa với model trả phí
import argparse
import json
import sys
import time

from hybrid_graphrag.config import G1_MAX_TOKENS, LLM_MODEL, VLLM_MODEL
from hybrid_graphrag.generation.llm_client import LLMError, call_gemini_json, get_usage
from hybrid_graphrag.generation.vllm_client import call_vllm_json, get_vllm_usage, vllm_server_ready
from hybrid_graphrag.knowledge_base.graph.build_g1 import load_chunks
from hybrid_graphrag.prompts.extraction import Extraction, build_extraction_prompt, clean_extraction

TOTAL_CHUNKS = 5454  # tổng chunk của 21 index (chạy tests.check_artifacts để xác nhận)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="Novel-25646", help="tên cuốn novel, hoặc 'medical'")
    ap.add_argument("ids", nargs="*", type=int, help="chỉ số chunk (mặc định: 21 64 76)")
    ap.add_argument("--extractor", choices=["gemini", "vllm"], default="gemini")
    ap.add_argument("--model", help="mặc định: VLLM_MODEL (vllm) hoặc LLM_MODEL (gemini)")
    ap.add_argument("--no-cache", action="store_true", help="Gemini: bỏ cache để đo token thật (tốn quota)")
    ap.add_argument("--quiet", action="store_true", help="chỉ in số liệu, không in JSON")
    ap.add_argument("--price-in", type=float)
    ap.add_argument("--price-out", type=float)
    a = ap.parse_args()

    use_vllm = a.extractor == "vllm"
    model = a.model or (VLLM_MODEL if use_vllm else LLM_MODEL)
    get = get_vllm_usage if use_vllm else get_usage
    if use_vllm and not vllm_server_ready():
        sys.exit("Không thấy server vLLM. Khởi động server trước.")

    domain = "medical" if a.source == "medical" else "novel"
    chunks = load_chunks(domain, a.source if domain == "novel" else None)
    ids = a.ids or [21, 64, 76]
    stats: dict = {}
    n_done, failed, times = 0, 0, []
    print(f"[{a.source}] {len(chunks)} chunk, extractor={a.extractor}, model={model}, thử {len(ids)} chunk")

    for i in ids:
        if not 0 <= i < len(chunks):
            print(f"[bỏ qua] chunk {i} ngoài khoảng 0..{len(chunks) - 1}")
            continue
        c = chunks[i]
        system, user = build_extraction_prompt(c["text"])
        before = get()
        t0 = time.time()
        try:
            if use_vllm:
                data = call_vllm_json(user, system, Extraction, max_output_tokens=G1_MAX_TOKENS, model=model)
            else:
                data = call_gemini_json(user, system, Extraction, model=model,
                                        max_output_tokens=G1_MAX_TOKENS, use_cache=not a.no_cache)
        except LLMError as e:
            failed += 1
            print(f"\n===== {c['chunk_id']}: LỖI {str(e)[:300]}")
            continue
        dt = time.time() - t0
        times.append(dt)
        after = get()
        used = {k: after[k] - before[k] for k in after}
        units = clean_extraction(data, stats)
        n_done += 1
        print(f"\n===== {c['chunk_id']} ({len(c['text'])} ký tự): {len(units)} unit | {dt:.1f}s | "
              f"{used['calls']} lượt, {used['prompt_tokens']} vào, {used['output_tokens']} ra =====")
        if not a.quiet:
            print(c["text"][:300] + "...")
            print(json.dumps(units, ensure_ascii=False, indent=2))

    print("\nTHỐNG KÊ:", stats)
    if stats.get("units"):
        print(f"Trung bình mỗi unit: {stats['entities'] / stats['units']:.1f} entity, "
              f"{stats['relations'] / stats['units']:.1f} quan hệ")
        dropped = sum(v for k, v in stats.items() if k.startswith("drop_"))
        print(f"Số mục bị lọc: {dropped} (so với {stats['relations']} quan hệ giữ lại)")
    if failed:
        print(f"Số chunk lỗi: {failed}/{len(ids)}")
    if times:
        print(f"Thời gian TB: {sum(times) / len(times):.1f}s/chunk (tuần tự; tốc độ thật khi chạy song song nhanh hơn nhiều)")

    u = get()
    print("USAGE:", u)
    if n_done and u["calls"] == 0:
        print("Không có lượt gọi thật (kết quả lấy từ cache). Thêm --no-cache hoặc chọn chunk khác.")
    elif u["calls"]:
        ptok, otok = u["prompt_tokens"] / u["calls"], u["output_tokens"] / u["calls"]
        print(f"\nTrung bình mỗi lượt gọi: {ptok:,.0f} token vào, {otok:,.0f} token ra "
              f"({u['calls']} lượt cho {n_done} chunk; gồm cả lượt thử lại)")
        tin, tout = TOTAL_CHUNKS * ptok, TOTAL_CHUNKS * otok
        print(f"Ước lượng G1 cho {TOTAL_CHUNKS:,} chunk: {tin / 1e6:.1f} triệu token vào, {tout / 1e6:.1f} triệu token ra")
        if a.price_in is not None and a.price_out is not None:
            print(f"Chi phí ước lượng: {tin / 1e6 * a.price_in + tout / 1e6 * a.price_out:,.1f} USD")
        print("Lưu ý: mẫu càng ít chunk càng kém tin cậy, nên đo trên 5-10 chunk có độ dài khác nhau.")


if __name__ == "__main__":
    main()