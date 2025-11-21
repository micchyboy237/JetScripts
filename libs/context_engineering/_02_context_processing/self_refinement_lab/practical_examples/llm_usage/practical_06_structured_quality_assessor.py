"""
PRACTICAL 6: Replace heuristic assessor with LLM structured output
Uses LlamacppLLM.chat_structured() → perfect parsing, no regex hell
"""
import os
from pydantic import BaseModel
from typing import List, Literal
from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    create_example_dir, get_logger, ProductionRefinementSystem, save_numpy, save_json
)
from jet.file.utils import save_file

class QualityAssessment(BaseModel):
    relevance: float  # 0-1
    coherence: float
    completeness: float
    clarity: float
    factuality: float
    overall: float
    issues: List[str]
    confidence: Literal["high", "medium", "low"]

def practical_06_structured_quality_assessor(
    query: str,
    context: str,
    llm: LlamacppLLM,
) -> QualityAssessment:
    example_dir = create_example_dir("practical_05_llm_critique_refine")
    logger = get_logger("llm_critic", example_dir)

    messages = [{
        "role": "system",
        "content": "You are an expert context quality assessor. Output only valid JSON matching the schema."
    }, {
        "role": "user",
        "content": f"Query: {query}\n\nContext:\n{context}"
    }]
    save_file(messages, os.path.join(example_dir, "messages.json"))

    assessment = llm.chat_structured(messages, response_model=QualityAssessment, temperature=0.0)
    save_file(assessment, os.path.join(example_dir, "assessment.json"))
    return assessment

if __name__ == "__main__":
    from jet.adapters.llama_cpp.llm import LlamacppLLM

    llm = LlamacppLLM(model="qwen3-instruct-2507:4b")

    assessment = practical_06_structured_quality_assessor(
        query="What are the main causes of LLM hallucinations?",
        context="""LLMs hallucinate because they are trained on internet data which contains contradictions. 
        Also, they don't have real understanding, just pattern matching. 
        Bad retrieval context makes it worse. 
        Temperature too high increases creativity and errors. 
        Context engineering helps a lot.""",
        llm=llm,
    )

    print("Structured Quality Assessment:")
    print(assessment.model_dump_json(indent=2))
