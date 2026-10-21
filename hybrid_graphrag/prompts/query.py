import re

from pydantic import BaseModel


class QueryEntities(BaseModel):
    entities: list[str]


SYSTEM_PROMPT = """You extract the key entities from a question so they can be matched against the nodes of a knowledge graph.

Return one flat list:
- Include named entities (people, organizations, places, titles, times) and the key concepts or objects the question is about.
- Write them in UPPERCASE, in the form they appear in the question.
- Skip question words and vague terms ("WHAT", "HOW", "WAY", "PROCESS", "PERSON"). Keep the list short: at most 8 items.
- If you are confident about the domain, you may add up to 2 closely related terms: the full name of an abbreviation in the question, or the abbreviation of a full name (for example "BCC" and "BASAL CELL CARCINOMA"). Never add terms that are only loosely related.
- If unsure, extract only what is literally in the question."""


def build_query_prompt(question: str) -> tuple[str, str]:
    return SYSTEM_PROMPT, f"Question: {question}"


def clean_entities(data: dict) -> list[str]:
    ents = (re.sub(r"\s+", " ", e).strip().upper() for e in data.get("entities", []))
    return list(dict.fromkeys(e for e in ents if e))