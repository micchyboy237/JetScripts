from haystack import Pipeline
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import EmbeddingRetriever, MarkdownConverter, PreProcessor
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
from jet.logger import CustomLogger
from readmedocs_fetcher_haystack import ReadmeDocsFetcher
from weaviate.embedded import EmbeddedOptions
import os
import shutil
import weaviate


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

# !pip install farm-haystack[weaviate,inference,file-conversion,preprocessing]

# !pip install readmedocs-fetcher-haystack


client = weaviate.Client(
  embedded_options=weaviate.embedded.EmbeddedOptions()
)


document_store = WeaviateDocumentStore(port=6666)

# from getpass import getpass

# readme_api_key = getpass("Enter ReadMe API key:")


converter = MarkdownConverter(remove_code_snippets=False)
readme_fetcher = ReadmeDocsFetcher(api_key=readme_api_key, markdown_converter=converter, base_url="https://docs.haystack.deepset.ai")
embedder = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
preprocessor = PreProcessor()


indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=readme_fetcher, name="ReadmeFetcher", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="Preprocessor", inputs=["ReadmeFetcher"])
indexing_pipeline.add_node(component=embedder, name="Embedder", inputs=["Preprocessor"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Embedder"])
indexing_pipeline.run()


answer_with_references_prompt = PromptTemplate(prompt = "You will be provided some conetent from technical documentation, where each paragraph is followed by the URL that it appears in. Answer the query based on the provided Documentation Content. Your answer should reference the URLs that it was generated from. Documentation Content: {join(documents, delimiter=new_line, pattern='---'+new_line+'$content'+new_line+'URL: $url', str_replace={new_line: ' ', '[': '(', ']': ')'})}\nQuery: {query}\nAnswer:", output_parser=AnswerParser())

# from getpass import getpass

# api_key = getpass("Enter OllamaFunctionCalling API key:")

prompt_node = PromptNode(model_name_or_path="gpt-4", api_key=api_key, default_prompt_template=answer_with_references_prompt, max_length=500)

pipeline = Pipeline()
pipeline.add_node(component = embedder, name = "Retriever", inputs = ["Query"])
pipeline.add_node(component = prompt_node, name = "GPT-4", inputs=["Retriever"])

def query(query:str):
  result = pipeline.run(query, params = {"Retriever": {"top_k": 5}})
  logger.debug(result['answers'][0].answer)
  return result

result = query("What are the optional installations of Haystack?")

logger.debug(result['answers'][0].meta['prompt'])

logger.info("\n\n[DONE]", bright=True)