from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.core import VectorStoreIndex
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import download_loader
import time
import os
os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"
loader = WikipediaReader()
documents = loader.load_data(pages=["Berlin"])
index = VectorStoreIndex.from_documents(documents)
print("Without optimization")
start_time = time.time()
query_engine = index.as_query_engine()
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))
print("With optimization")
start_time = time.time()
query_engine = index.as_query_engine(
    node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)]
)
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))
print("Alternate optimization cutoff")
start_time = time.time()
query_engine = index.as_query_engine(
    node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.7)]
)
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))
