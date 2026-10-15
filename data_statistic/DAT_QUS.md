# Ghi chú thống kê dataset – Hybrid GraphRAG

Nguồn: kết quả chạy `python -m data_statistic.dataset_stat` và `python -m data_statistic.question_stat`
(file JSON: `data_statistic/dataset_stats.json`, `data_statistic/question_stats.json`).

> Chunk/token là **ước lượng thô** (chunk size 1200, overlap 100, 4 ký tự ≈ 1 token, overhead prompt G1 ≈ 800 token/chunk).

---

## 1. Corpus

| | Medical | Novel | Tổng |
|---|---|---|---|
| Dung lượng | 1.01 MB | 4.62 MB | 5.63 MB |
| Số document | 1 | 20 | 21 |
| Fields | `context`, `corpus_name` | `context`, `corpus_name` | |
| Ký tự | 1,052,159 | 4,819,610 | |
| Từ | 174,610 | 839,608 | |
| Câu | 11,449 (TB 92 ký tự) | 43,829 (TB 110 ký tự) | |
| Token ước lượng | ~263K | ~1.2M | |
| Chunk (1200/100) | ~957 | ~4,389 | ~5,346 |
| G1: số lần gọi LLM | ~957 | ~4,389 | ~5,346 |
| G1: token đầu vào | ~1.03M | ~4.72M | ~5.74M |

Novel theo từng cuốn: mean 240,980 ký tự, median 240,891, min 92,796, max 450,081, std 93,834.

### Novel – từng document (sắp theo số từ)

`*` = subset chạy thử (đề xuất)

| | Index | Name | Từ | Chunk |
|---|---|---|---|---|
| * | 11 | Novel-25646 | 18,042 | 90 |
| * | 18 | Novel-4128 | 18,244 | 85 |
| * | 17 | Novel-40700 | 23,593 | 126 |
| | 10 | Novel-47558 | 25,728 | 131 |
| | 5 | Novel-29973 | 27,716 | 150 |
| | 0 | Novel-30752 | 29,099 | 153 |
| | 15 | Novel-54537 | 31,751 | 174 |
| | 2 | Novel-58553 | 33,786 | 175 |
| | 9 | Novel-8559 | 39,911 | 206 |
| | 19 | Novel-44557 | 41,843 | 216 |
| | 6 | Novel-10146 | 42,433 | 222 |
| | 12 | Novel-5956 | 47,802 | 233 |
| | 14 | Novel-41603 | 48,416 | 243 |
| | 16 | Novel-10356 | 52,029 | 281 |
| | 8 | Novel-10321 | 53,841 | 288 |
| | 7 | Novel-47676 | 53,925 | 277 |
| | 3 | Novel-10762 | 58,359 | 312 |
| | 1 | Novel-51410 | 60,635 | 304 |
| | 13 | Novel-2544 | 60,973 | 313 |
| | 4 | Novel-26183 | 71,482 | 410 |

---

## 2. Câu hỏi

| | Medical | Novel |
|---|---|---|
| Tổng số câu | 2,062 | 2,010 |
| Fields | `answer`, `evidence`, `evidence_relations`, `id`, `question`, `question_type`, `source` | `answer`, `evidence`, `evidence_triple`, `id`, `question`, `question_type`, `source` |
| `source` | 1 giá trị (`Medical`) | 20 giá trị (trùng `corpus_name`) |

### Medical

| Loại | n | Answer từ (med / p90 / max) | Answer câu (med / max) | Evidence TB |
|---|---|---|---|---|
| Fact Retrieval | 1098 | 10 / 15.1 / 45 | 1 / 2 | 2.3 |
| Complex Reasoning | 509 | 17 / 23.0 / 32 | 1 / 2 | 3.5 |
| Contextual Summarize | 289 | 76 / 146.0 / 216 | 3 / 8 | 6.1 |
| Creative Generation | 166 | 160.5 / 252.8 / 418 | 9 / 28 | 13.1 |

### Novel

| Loại | n | Answer từ (med / p90 / max) | Answer câu (med / max) | Evidence TB |
|---|---|---|---|---|
| Fact Retrieval | 971 | 6 / 11.0 / 147 | 1 / 11 | 1.4 |
| Complex Reasoning | 610 | 16 / 27.0 / 110 | 1 / 11 | 2.6 |
| Contextual Summarize | 362 | 93 / 129.7 / 172 | 3 / 12 | 3.5 |
| Creative Generation | 67 | 133 / 163.8 / 196 | 5 / 10 | 7.1 |

Novel – top 5 cuốn theo số câu hỏi: Novel-8559 (137), Novel-10356 (130), Novel-10146 (126), Novel-54537 (120), Novel-2544 (114).

---

## 3. Quyết định / đề xuất (chưa chốt, cần xác nhận khi code)

1. **Index theo từng cuốn novel**: `source` của câu hỏi trùng `corpus_name`, nên build FAISS + graph riêng cho từng cuốn, retrieval chỉ chạy trong cuốn của câu hỏi.
   ```
   artifacts/
   ├── medical/        (faiss, metadata.pkl, g1/g2/g3.gml)
   └── novel/<Novel-XXXX>/
   ```
2. **Subset chạy thử**: Novel-25646 (90 chunk) + Novel-4128 (85 chunk); thêm Novel-40700 (126 chunk) nếu thiếu câu Complex Reasoning / Creative Generation. Medical chỉ có 1 document → test trên novel trước, chạy medical đầy đủ một lần.
3. **Chunk size**: giữ 1200/100 (tăng lên 2000 chỉ khi quota không đủ).
4. **`top_k` và output theo loại câu hỏi** (điểm xuất phát để tune):

   | Loại | `top_k` | `max_output_tokens` |
   |---|---|---|
   | Fact Retrieval | 5–10 | ~128 |
   | Complex Reasoning | 10 | ~256 |
   | Contextual Summarize | 10–15 | ~512 |
   | Creative Generation | 15–20 | ~1024 |

   - `dual_search` nhận `top_k` như tham số, không hardcode.
   - Fact Retrieval: prompt phải yêu cầu trả lời ngắn (answer chỉ 6–10 từ), nếu không Rouge-L sẽ thấp dù đúng nội dung.
5. **Đánh giá**:
   - Báo cáo theo từng loại câu hỏi, không gộp.
   - Khi tune dùng mẫu phân tầng 30–50 câu/loại (không lấy N câu đầu).
   - Answer Accuracy là metric chính cho Fact Retrieval; Rouge-L khắt khe với answer ngắn.
6. **G1**: ràng buộc chính là số lần gọi LLM (~5,346), cần cache theo chunk và resume được. Làm medical (~957 chunk) và novel theo từng cuốn.

---

## 4. Việc cần làm tiếp

- [ ] Đếm số câu hỏi theo loại cho Novel-25646, Novel-4128, Novel-40700 để chốt subset
- [ ] Viết `config/settings.py` (đường dẫn, `CHUNK_SIZE/OVERLAP`, tham số retrieval theo loại, cấu trúc `artifacts/`)
- [ ] Viết `generation/llm_client.py` (xoay vòng key, retry, rate-limit, cache)
- [ ] Kiểm tra RPD thực tế của model Gemini trong Google AI Studio để ước lượng số ngày build G1

---

## 5. Ghi chú thêm

<!-- thêm ghi chú của bạn ở đây -->