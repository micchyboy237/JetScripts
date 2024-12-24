```python
# !pip install llama-index

from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader

import nest_asyncio

nest_asyncio.apply()
# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents)
query = "What did the author do growing up?"
base_query_engine = index.as_query_engine()
response = base_query_engine.query(query)
print(response)
from llama_index.core.query_engine import RetryQueryEngine
from llama_index.core.evaluation import RelevancyEvaluator

query_response_evaluator = RelevancyEvaluator()
retry_query_engine = RetryQueryEngine(
    base_query_engine, query_response_evaluator
)
retry_response = retry_query_engine.query(query)
print(retry_response)
from llama_index.core.query_engine import RetrySourceQueryEngine

retry_source_query_engine = RetrySourceQueryEngine(
    base_query_engine, query_response_evaluator
)
retry_source_response = retry_source_query_engine.query(query)
print(retry_source_response)
from llama_index.core.evaluation import GuidelineEvaluator
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from llama_index.core import Response
from llama_index.core.indices.query.query_transform.feedback_transform import (
    FeedbackQueryTransformation,
)
from llama_index.core.query_engine import RetryGuidelineQueryEngine

guideline_eval = GuidelineEvaluator(
    guidelines=DEFAULT_GUIDELINES
    + "\nThe response should not be overly long.\n"
    "The response should try to summarize where possible.\n"
)  # just for example
typed_response = (
    response if isinstance(response, Response) else response.get_response()
)
eval = guideline_eval.evaluate_response(query, typed_response)
print(f"Guideline eval evaluation result: {eval.feedback}")

feedback_query_transform = FeedbackQueryTransformation(resynthesize_query=True)
transformed_query = feedback_query_transform.run(query, {"evaluation": eval})
print(f"Transformed query: {transformed_query.query_str}")
retry_guideline_query_engine = RetryGuidelineQueryEngine(
    base_query_engine, guideline_eval, resynthesize_query=True
)
retry_guideline_response = retry_guideline_query_engine.query(query)
print(retry_guideline_response)
```