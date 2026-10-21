import re

from pydantic import BaseModel


class Relation(BaseModel):
    source: str
    relation: str
    target: str


class SemanticUnit(BaseModel):
    summary: str
    entities: list[str]
    relations: list[Relation]


class Extraction(BaseModel):
    semantic_units: list[SemanticUnit]


SYSTEM_PROMPT = """You build a knowledge graph from text. Given a passage, split it into semantic units and extract entities and relations from each unit.

A semantic unit is a self-contained piece of the passage about one event, activity, fact or explanation. A unit usually spans 2-5 consecutive sentences: merge sentences that belong to the same idea, and never make a unit out of one short sentence unless it states a distinct fact. Units follow the order of the passage and together cover all its factual content.

For each semantic unit return:
1. summary: a faithful paraphrase that keeps all crucial details (who, what, where, when, numbers, causes). It must make sense on its own: name the topic explicitly, replace pronouns ("he", "she", "it", "they") with the names they refer to, and never write references such as "the first story", "this" or "the above".
2. entities: the entities found in the ORIGINAL text of the unit.
3. relations: facts connecting two of those entities, as {source, relation, target}.

Rules:
- An entity is a concrete or named thing: a person, character, organization, place, time, object, animal, body part, substance, disease, treatment, or a key concept. An entity is never an adjective, adverb, verb or state word (for example WELL, ABOUT, CARELESS, HURT, DOING). Put such words inside the relation text instead.
- Do not use generic placeholders as entities: PEOPLE, MEN, MAN, PERSON, ONE, THING, WAY. Attach the fact to the specific entity that causes or undergoes it. If no specific entity exists, skip the relation; the summary still keeps the fact.
- Only for an individual character in a story or anecdote who is introduced as "a boy" or "a man", give a short distinguishing name such as "BOY WHO LOOKED AT THE SUN" and use it for the whole passage. Never do this for groups, general statements or hypothetical people ("men who drink", "a person who is drunk", "a man who loses control"): for those, skip the relation and let the summary keep the fact, or attach the fact to the cause or the body part involved. Entity names are short noun phrases of at most 5 words.
- Entity names are UPPERCASE, one consistent name per entity across the whole passage, preferring the fullest form in the text (for example "DR. EMILY ROBERTS", not "ROBERTS"). No duplicates within a unit.
- Times: keep the precision stated in the text and never fill in missing parts: "2024-09" for September 2024, "2024" for a year alone, "1990S" for a decade. Keep relative times as written (for example "LAST WINTER").
- relation is a short lowercase phrase. Read-aloud test: "SOURCE relation TARGET" must be a complete statement that the passage supports. Qualifiers (manner, reason) belong in the relation text, for example {"source": "ALCOHOL", "relation": "injures and keeps from working properly", "target": "BRAIN"}.
- source and target must both appear in that unit's entities list, and each must be a single entity. The relation text must not mention any other entity of the unit; if a fact involves three entities, split it into two relations.
- Use only information stated in the passage. Do not add outside knowledge.
- If the passage has no factual content (headings, page numbers, boilerplate), return an empty list of semantic units.

Example passage:
In September 2024, Dr. Emily Roberts traveled to Paris to attend the International Conference on Renewable Energy, where she presented her research on solar panel efficiency. Meanwhile, her colleague Dr. John Miller documented new species in the Amazon Rainforest and observed how deforestation harms local wildlife.

Example output:
{"semantic_units": [
  {"summary": "In September 2024, Dr. Emily Roberts traveled to Paris, attended the International Conference on Renewable Energy and presented her research on solar panel efficiency.",
   "entities": ["DR. EMILY ROBERTS", "2024-09", "PARIS", "INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY", "SOLAR PANEL EFFICIENCY"],
   "relations": [
     {"source": "DR. EMILY ROBERTS", "relation": "traveled to", "target": "PARIS"},
     {"source": "DR. EMILY ROBERTS", "relation": "attended", "target": "INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY"},
     {"source": "DR. EMILY ROBERTS", "relation": "presented research on", "target": "SOLAR PANEL EFFICIENCY"},
     {"source": "INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY", "relation": "was held in", "target": "2024-09"}]},
  {"summary": "Dr. John Miller, a colleague of Dr. Emily Roberts, documented new species in the Amazon Rainforest and observed how deforestation harms local wildlife.",
   "entities": ["DR. JOHN MILLER", "DR. EMILY ROBERTS", "NEW SPECIES", "AMAZON RAINFOREST", "DEFORESTATION", "LOCAL WILDLIFE"],
   "relations": [
     {"source": "DR. JOHN MILLER", "relation": "is a colleague of", "target": "DR. EMILY ROBERTS"},
     {"source": "DR. JOHN MILLER", "relation": "documented", "target": "NEW SPECIES"},
     {"source": "NEW SPECIES", "relation": "were found in", "target": "AMAZON RAINFOREST"},
     {"source": "DR. JOHN MILLER", "relation": "observed the effects of", "target": "DEFORESTATION"},
     {"source": "DEFORESTATION", "relation": "harms", "target": "LOCAL WILDLIFE"}]}
]}"""


def build_extraction_prompt(text: str) -> tuple[str, str]:
    """Trả về (system_prompt, user_prompt). Phần cố định nằm ở system để dễ được cache phía Gemini."""
    return SYSTEM_PROMPT, f"Passage:\n{text}"


GENERIC_ENTITIES = {
    "PEOPLE", "PERSON", "PERSONS", "MEN", "MAN", "WOMEN", "WOMAN", "ONE", "SOMEONE",
    "THING", "THINGS", "WAY", "WELL", "ABOUT", "DOING",
}  # mở rộng dần dựa trên data_statistic/g1_stat.py (entity xuất hiện nhiều nhưng vô nghĩa)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().upper()


def _mentions(text: str, entity: str) -> bool:
    return re.search(rf"\b{re.escape(entity.lower())}s?\b", text.lower()) is not None


def _plural_map(ents: list[str]) -> dict[str, str]:
    """MUSCLES -> MUSCLE nếu cả hai cùng có trong unit."""
    s = set(ents)
    return {e: e[:-1] for e in ents if e.endswith("S") and e[:-1] in s}


def clean_extraction(data: dict, stats: dict | None = None) -> list[dict]:
    """Chuẩn hóa kết quả LLM; stats (nếu có) đếm số mục bị loại để theo dõi chất lượng prompt."""
    stats = stats if stats is not None else {}

    def bump(key: str) -> None:
        stats[key] = stats.get(key, 0) + 1

    units = []
    for u in data.get("semantic_units", []):
        summary = u["summary"].strip()
        if not summary:
            continue
        ents = list(dict.fromkeys(e for e in map(_norm, u["entities"]) if e))
        n_before = len(ents)
        ents = [e for e in ents if e not in GENERIC_ENTITIES]
        stats["drop_generic_entity"] = stats.get("drop_generic_entity", 0) + n_before - len(ents)

        pm = _plural_map(ents)
        ents = list(dict.fromkeys(pm.get(e, e) for e in ents))

        rels, seen = [], set()
        for r in u["relations"]:
            s, t = _norm(r["source"]), _norm(r["target"])
            s, t = pm.get(s, s), pm.get(t, t)
            rel = re.sub(r"\s+the$", "", r["relation"].strip().lower())
            if not (s and t and rel) or s == t:
                bump("drop_empty_or_self")
                continue
            if s in GENERIC_ENTITIES or t in GENERIC_ENTITIES:
                bump("drop_generic_relation")
                continue
            if any(_mentions(rel, x) for x in ents if x not in (s, t) and len(x) > 2):
                bump("drop_third_entity")  # relation chứa entity thứ ba, thông tin vẫn còn trong summary
                continue
            if (s, rel, t) in seen:
                continue
            seen.add((s, rel, t))
            rels.append({"source": s, "relation": rel, "target": t})
            for x in (s, t):
                if x not in ents:
                    ents.append(x)
        stats["entities"] = stats.get("entities", 0) + len(ents)
        stats["relations"] = stats.get("relations", 0) + len(rels)
        stats["units"] = stats.get("units", 0) + 1
        units.append({"summary": summary, "entities": ents, "relations": rels})
    return units