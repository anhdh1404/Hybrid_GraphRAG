import json
from pathlib import Path
from typing import Any

from hybrid_graphrag.config import CORPUS_DIR, CORPUS_FILES, QUESTIONS_DIR, QUESTION_FILES

VALID_DOMAINS = tuple(CORPUS_FILES)  # ("medical", "novel")


def _check_domain(domain: str) -> str:
    d = domain.lower()
    if d not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of: {list(VALID_DOMAINS)}")
    return d


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_corpus(domain: str, source: str | None = None) -> list[dict[str, str]]:
    """[{'corpus_name', 'context'}, ...]. medical: 1 phần tử; novel: 20.
    source: chỉ lấy một cuốn (so khớp corpus_name)."""
    data = _read_json(CORPUS_DIR / CORPUS_FILES[_check_domain(domain)])
    docs = data if isinstance(data, list) else [data]
    if source is not None:
        docs = [d for d in docs if d["corpus_name"] == source]
        if not docs:
            raise ValueError(f"Không có corpus_name='{source}' trong {domain}")
    return docs


def load_questions(
    domain: str, source: str | None = None, question_type: str | None = None
) -> list[dict[str, Any]]:
    """Lọc tùy chọn theo cuốn (source) và loại câu hỏi (question_type)."""
    items = _read_json(QUESTIONS_DIR / QUESTION_FILES[_check_domain(domain)])
    if source is not None:
        items = [q for q in items if q["source"] == source]
    if question_type is not None:
        items = [q for q in items if q["question_type"] == question_type]
    return items


def load_dataset(domain: str, source: str | None = None) -> dict[str, list]:
    return {
        "corpus": load_corpus(domain, source),
        "questions": load_questions(domain, source),
    }