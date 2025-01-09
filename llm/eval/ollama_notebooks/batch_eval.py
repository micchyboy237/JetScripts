import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
from jet.llm.ollama.base import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response
from jet.llm.ollama import initialize_ollama_settings
import os
import nest_asyncio
```python
nest_asyncio.apply()
os.environ["OPENAI_API_KEY"] = "sk-..."
initialize_ollama_settings()
pd.set_option("display.max_colwidth", 0)
ollama = Ollama(temperature=0, model="llama3.1")
faithfulness_ollama = FaithfulnessEvaluator(llm=ollama)
relevancy_ollama = RelevancyEvaluator(llm=ollama)
correctness_ollama = CorrectnessEvaluator(llm=ollama)
documents = SimpleDirectoryReader("./test_wiki_data/").load_data()
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[SentenceSplitter(chunk_size=512)]
)
dataset_generator = DatasetGenerator.from_documents(documents, llm=ollama)
qas = dataset_generator.generate_dataset_from_nodes(num=3)
runner = BatchEvalRunner(
    {"faithfulness": faithfulness_ollama, "relevancy": relevancy_ollama},
    workers=8,
)
eval_results = await runner.aevaluate_queries(
    vector_index.as_query_engine(llm=ollama), queries=qas.questions
)
print(len([qr for qr in qas.qr_pairs]))
print(eval_results.keys())
print(eval_results["faithfulness"][0].dict().keys())
print(eval_results["faithfulness"][0].passing)
print(eval_results["faithfulness"][0].response)
print(eval_results["faithfulness"][0].contexts)


def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score


score = get_eval_results("faithfulness", eval_results)
score = get_eval_results("relevancy", eval_results)
```
