!pip install -q -q llama-index
!pip install -U -q deepeval!deepeval loginfrom llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
rag_application = index.as_query_engine()
from deepeval.integrations.llamaindex import DeepEvalFaithfulnessEvaluator

user_input = "What is LlamaIndex?"

response_object = rag_application.query(user_input)

evaluator = DeepEvalFaithfulnessEvaluator()
evaluation_result = evaluator.evaluate_response(
    query=user_input, response=response_object
)
print(evaluation_result)
