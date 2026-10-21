SYSTEM_PROMPT = (
    "You answer questions using only the retrieved context. "
    "Do not use outside knowledge and do not add details the context does not support. "
    "Answer directly, without phrases like 'according to the context' and without citing passage numbers. "
    "If the context does not contain enough information, say in one short sentence what is missing instead of guessing."
)

# Độ dài đích lấy từ thống kê answer (median số từ theo loại câu hỏi)
LENGTH_GUIDE = {
    "Fact Retrieval": "Answer in one short sentence (about 10 words), no explanation.",
    "Complex Reasoning": "Answer in one or two sentences (about 20 words).",
    "Contextual Summarize": "Answer in one paragraph of about 80-100 words.",
    "Creative Generation": "Answer in about 150 words, well organized.",
}
DEFAULT_GUIDE = "Answer concisely."


def build_prompt(
    question: str,
    chunks: list[dict],
    question_type: str | None = None,
    facts: list[str] | None = None,   # quan hệ từ đồ thị (dùng ở bước hybrid)
) -> tuple[str, str]:
    """Trả về (system_prompt, user_prompt)."""
    parts = ["Context passages:\n" + "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))]
    if facts:
        parts.append("Related facts from the knowledge graph:\n" + "\n".join(f"- {f}" for f in facts))
    guide = LENGTH_GUIDE.get(question_type, DEFAULT_GUIDE)
    parts.append(f"Question: {question}\n\n{guide}\nAnswer:")
    return SYSTEM_PROMPT, "\n\n".join(parts)