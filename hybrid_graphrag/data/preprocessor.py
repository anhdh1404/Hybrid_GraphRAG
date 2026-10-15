import re

from hybrid_graphrag.config import CHUNK_OVERLAP, CHUNK_SIZE

# Tách sau . ! ? (kể cả khi theo sau là dấu đóng ngoặc/nháy), không làm mất dấu đóng
_SENT_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)]))\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    """Cắt câu quá dài theo ký tự, ưu tiên ranh giới từ."""
    parts = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size)
        if cut < size * 0.5:
            cut = size
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def _tail(sentences: list[str], overlap: int) -> list[str]:
    """Các câu cuối có tổng độ dài <= overlap, dùng làm phần overlap của chunk kế."""
    keep, total = [], 0
    for s in reversed(sentences):
        new_total = total + len(s) + (1 if keep else 0)
        if new_total > overlap:
            break
        keep.append(s)
        total = new_total
    return keep[::-1]


def create_chunks(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Gom câu thành chunk <= chunk_size ký tự; chunk sau lặp lại vài câu cuối của chunk trước."""
    sentences: list[str] = []
    for s in split_sentences(text):
        sentences.extend(_hard_split(s, chunk_size) if len(s) > chunk_size else [s])

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        if cur and cur_len + 1 + len(s) > chunk_size:
            chunks.append(" ".join(cur))
            cur = _tail(cur, overlap)
            cur_len = len(" ".join(cur))
            if cur and cur_len + 1 + len(s) > chunk_size:  # overlap + câu mới không vừa
                cur, cur_len = [], 0
        cur_len = cur_len + 1 + len(s) if cur else len(s)
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def make_chunks(doc: dict) -> list[dict]:
    """doc = {'corpus_name', 'context'} -> list chunk có id ổn định (khóa cache cho G1)."""
    name = doc["corpus_name"]
    return [
        {"chunk_id": f"{name}_{i:05d}", "source": name, "index": i, "text": t}
        for i, t in enumerate(create_chunks(doc["context"]))
    ]


if __name__ == "__main__":
    import sys

    from hybrid_graphrag.data.loader import load_corpus

    sys.stdout.reconfigure(encoding="utf-8")
    for domain in ("medical", "novel"):
        chunks = [c for d in load_corpus(domain) for c in make_chunks(d)]
        lens = [len(c["text"]) for c in chunks]
        print(f"{domain}: {len(chunks):,} chunk | TB {sum(lens)/len(lens):.0f} ký tự | max {max(lens)}")
        assert max(lens) <= CHUNK_SIZE, "có chunk vượt CHUNK_SIZE"
    print("OK")