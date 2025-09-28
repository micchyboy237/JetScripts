import os
import shutil

from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from jet.adapters.haystack.deepeval.ollama_model import OllamaModel
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

model = OllamaModel(
    model="deepseek-r1:1.5b-qwen-distill-q4_K_M",
    base_url="http://localhost:11434",
    temperature=0
)

contextual_relevancy_metric = ContextualRelevancyMetric(
    threshold=0.5,  # Optional: Minimum passing score (default: 0.5)
    model=model,  # Optional: LLM model (default: "gpt-4o")
    include_reason=True,  # Optional: Include explanation (default: False)
    # Additional optional params: strict_mode (bool), use_gpt_vision (bool for multimodal),
    # async_mode (bool), cache (bool)
)

# Create a test case for RAG retrieval evaluation
test_case = LLMTestCase(
    input="What is the capital of France?",  # User query
    retrieval_context=["Paris is the capital and largest city of France.", "France is in Europe."]  # Retrieved documents
)

# Measure the score
contextual_relevancy_metric.measure(test_case)
print(f"Score: {contextual_relevancy_metric.score}")  # e.g., 0.85 (0-1 scale)
print(f"Reason: {contextual_relevancy_metric.reason}")  # Explanation if include_reason=True
# print(f"Passing: {contextual_relevancy_metric.is_passing}")  # True if score >= threshold
save_file(contextual_relevancy_metric, f"{OUTPUT_DIR}/contextual_relevancy_metric.json")
