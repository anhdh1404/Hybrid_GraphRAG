SYSTEM_PROMPT = """You write short, factual summaries for a retrieval system. Given a subject (an entity or a group of related entities) together with the passages and relations that mention it, write a summary that will help answer questions about it later.

Rules:
- Use only the information provided. Do not add outside knowledge and do not speculate.
- State the subject's key attributes and events and its most important connections to other entities. Keep names, numbers, dates and causes exact.
- Plain, neutral prose in a single paragraph. No headings, no bullet lists, no filler such as "this summary" or "the text says".
- At most 200 words; shorter when the input is limited.
- Output only the summary."""


def build_summary_prompt(
    subject: str,
    semantic_units: list[str],
    relationships: list[tuple[str, str, str]],
    max_units: int = 30,
    max_rels: int = 60,
) -> tuple[str, str]:
    units = "\n".join(f"- {u}" for u in semantic_units[:max_units])
    rels = "\n".join(f"- {s} | {r} | {t}" for s, r, t in relationships[:max_rels])
    user = (
        f"Subject: {subject}\n\n"
        f"Related passages:\n{units}\n\n"
        f"Related relations (source | relation | target):\n{rels}"
    )
    return SYSTEM_PROMPT, user