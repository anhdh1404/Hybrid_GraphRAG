# Bốn metric theo cấu hình của GraphRAG-Bench, judge là Gemini
# (mỗi metric = 1 lần gọi LLM; ROUGE-L không gọi LLM).
#   Fact Retrieval, Complex Reasoning : answer_correctness + ROUGE-L
#   Contextual Summarize              : answer_correctness + coverage
#   Creative Generation               : answer_correctness + coverage + faithfulness
import math
import statistics

from pydantic import BaseModel
from rouge_score import rouge_scorer

from hybrid_graphrag.generation.llm_client import call_gemini_json

METRICS_BY_TYPE = {
    "Fact Retrieval": ["rouge_l", "answer_correctness"],
    "Complex Reasoning": ["rouge_l", "answer_correctness"],
    "Contextual Summarize": ["answer_correctness", "coverage"],
    "Creative Generation": ["answer_correctness", "coverage", "faithfulness"],
}
JUDGE_TOKENS = 3000
MAX_CONTEXT_CHARS = 60_000
FACT_WEIGHT, SIM_WEIGHT = 0.75, 0.25  # answer correctness = 0.75 * F-beta + 0.25 * tương đồng ngữ nghĩa
BETA = 1.0

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def mean(values):
    v = [x for x in values if x is not None and not math.isnan(x)]
    return sum(v) / len(v) if v else None


def std(values):
    v = [x for x in values if x is not None and not math.isnan(x)]
    return statistics.pstdev(v) if len(v) > 1 else 0.0


# ------------------------------------------------------------------ ROUGE-L
def rouge_l(answer: str, gold: str) -> float:
    if not answer.strip() or not gold.strip():
        return 0.0
    return _ROUGE.score(gold, answer)["rougeL"].fmeasure


# --------------------------------------------------------- answer correctness
class CorrectnessJudgement(BaseModel):
    tp: list[str]  # phát biểu trong câu trả lời, được đáp án chuẩn ủng hộ
    fp: list[str]  # phát biểu trong câu trả lời, không được đáp án chuẩn ủng hộ
    fn: list[str]  # phát biểu trong đáp án chuẩn, câu trả lời bỏ sót


_AC_SYSTEM = """You are a strict evaluator. Compare an answer with the ground-truth answer to a question.
Break both texts into short, self-contained, atomic statements, then classify:
- tp: statements in the answer that are present in, or directly supported by, the ground truth.
- fp: statements in the answer that are not supported by the ground truth (including contradictions).
- fn: statements in the ground truth that the answer is missing.
Judge only against the ground truth, not against outside knowledge. Ignore wording and style; compare meaning. Return JSON.

Example:
Question: What causes the seasons?
Ground truth: Seasons result from Earth's axial tilt. The tilt makes hemispheres receive different amounts of sunlight.
Answer: Seasons are caused by Earth's tilted axis. They happen because Earth moves closer to and farther from the sun.
Output: {"tp": ["Seasons are caused by Earth's tilted axis"], "fp": ["Seasons happen because Earth moves closer to and farther from the sun"], "fn": ["The tilt makes hemispheres receive different amounts of sunlight"]}"""


def fbeta(tp: int, fp: int, fn: int, beta: float = BETA) -> float:
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)


def semantic_similarity(a: str, b: str) -> float:
    """Cosine giữa hai văn bản, đưa về thang [0, 1] bằng (cos + 1) / 2 như bản gốc."""
    from hybrid_graphrag.knowledge_base.embeddings import embed_texts

    v = embed_texts([a, b], show_progress=False)
    return (float(v[0] @ v[1]) + 1) / 2


def answer_correctness(question: str, answer: str, gold: str, model: str, use_similarity: bool = True):
    user = f"Question: {question}\n\nGround truth: {gold}\n\nAnswer: {answer}"
    j = call_gemini_json(user, _AC_SYSTEM, CorrectnessJudgement, model=model, max_output_tokens=JUDGE_TOKENS)
    tp, fp, fn = len(j["tp"]), len(j["fp"]), len(j["fn"])
    if tp + fp + fn == 0:
        return None  # judge không tách được phát biểu nào: bỏ qua mẫu này
    f = fbeta(tp, fp, fn)
    if not use_similarity:
        return f
    return FACT_WEIGHT * f + SIM_WEIGHT * semantic_similarity(answer, gold)


# ------------------------------------------------------------------- coverage
class FactCheck(BaseModel):
    fact: str
    covered: bool


class CoverageJudgement(BaseModel):
    facts: list[FactCheck]


_COV_SYSTEM = """List the key facts of the ground-truth answer that are needed to answer the question (at most 15, short and self-contained).
For each fact, set covered=true only if the response clearly states that fact or an equivalent one; otherwise false.
Judge only the response against the ground truth. Return JSON."""


def coverage_score(question: str, gold: str, answer: str, model: str):
    user = f"Question: {question}\n\nGround-truth answer: {gold}\n\nResponse: {answer}"
    j = call_gemini_json(user, _COV_SYSTEM, CoverageJudgement, model=model, max_output_tokens=JUDGE_TOKENS)
    facts = j["facts"]
    return sum(f["covered"] for f in facts) / len(facts) if facts else None


# ---------------------------------------------------------------- faithfulness
class ClaimCheck(BaseModel):
    claim: str
    supported: bool


class FaithfulnessJudgement(BaseModel):
    claims: list[ClaimCheck]


_FAITH_SYSTEM = """Extract the distinct factual claims made in the answer (at most 20, short and self-contained).
For each claim, set supported=true only if it can be inferred from the context passages; otherwise false.
Use only the context, not outside knowledge. Return JSON."""


def faithfulness_score(question: str, answer: str, contexts: list[str], model: str):
    ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))[:MAX_CONTEXT_CHARS]
    user = f"Question: {question}\n\nContext passages:\n{ctx}\n\nAnswer: {answer}"
    j = call_gemini_json(user, _FAITH_SYSTEM, FaithfulnessJudgement, model=model, max_output_tokens=JUDGE_TOKENS)
    claims = j["claims"]
    return sum(c["supported"] for c in claims) / len(claims) if claims else None


# ------------------------------------------------------------------- một mẫu
def evaluate_sample(rec: dict, model: str, use_similarity: bool = True, get_contexts=None) -> dict:
    """rec: bản ghi của run_baseline. get_contexts: hàm không tham số trả về list[str] (chỉ gọi khi cần faithfulness)."""
    q, ans, gold = rec["question"], rec["answer"], rec["gold_answer"]
    out: dict = {}
    for m in METRICS_BY_TYPE.get(rec["question_type"], ["answer_correctness"]):
        if m == "rouge_l":
            out[m] = rouge_l(ans, gold)
        elif m == "answer_correctness":
            out[m] = answer_correctness(q, ans, gold, model, use_similarity)
        elif m == "coverage":
            out[m] = coverage_score(q, gold, ans, model)
        elif m == "faithfulness":
            out[m] = faithfulness_score(q, ans, get_contexts() if get_contexts else [], model)
    return out