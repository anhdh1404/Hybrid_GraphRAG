import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# --- Đường dẫn ---
CORPUS_DIR = ROOT / "GraphRAG-Bench" / "Datasets" / "Corpus"
QUESTIONS_DIR = ROOT / "GraphRAG-Bench" / "Datasets" / "Questions"
ARTIFACTS_DIR = ROOT / "artifacts"

CORPUS_FILES = {"medical": "medical.json", "novel": "novel.json"}
QUESTION_FILES = {"medical": "medical_questions.json", "novel": "novel_questions.json"}

# --- Chunking ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

# --- Embedding ---
EMBEDDING_MODEL = "google/embeddinggemma-300m"
EMBEDDING_DIM = 768          # có thể giảm 512/256/128 (Matryoshka), cần build lại index
EMBEDDING_MAX_SEQ_LEN = 512  # khuyến nghị của model; chunk ~300 token nên đủ

# --- LLM ---
LLM_MODEL = "gemini-2.5-flash"

GEMINI_API_KEYS = list(dict.fromkeys(
    v for k, v in sorted(os.environ.items()) if k.startswith("GEMINI_API_KEY") and v
))

LLM_MAX_RETRIES = 12      # tổng số lần thử (tính cả lần đổi key)
LLM_MIN_INTERVAL = 0.0    # giây tối thiểu giữa 2 lần gọi; tăng lên nếu hay dính 429
LLM_TEMPERATURE = 0.0

# --- Retrieval: tham số mặc định và theo loại câu hỏi (điểm xuất phát để tune) ---
ALPHA = 0.5
PPR_ITERATIONS = 2
LAMBDA_WEIGHT = 0.5

RETRIEVAL_BY_TYPE = {
    "Fact Retrieval":       {"top_k": 8,  "max_output_tokens": 128},
    "Complex Reasoning":    {"top_k": 10, "max_output_tokens": 256},
    "Contextual Summarize": {"top_k": 12, "max_output_tokens": 512},
    "Creative Generation":  {"top_k": 18, "max_output_tokens": 1024},
}


def index_dir(domain: str, source: str | None = None) -> Path:
    """medical -> artifacts/medical ; novel -> artifacts/novel/<Novel-XXXX>"""
    path = ARTIFACTS_DIR / domain
    if domain == "novel" and source:
        path = path / source
    path.mkdir(parents=True, exist_ok=True)
    return path