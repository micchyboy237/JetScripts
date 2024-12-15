%pip install llama-index-llms-openai!pip install llama-indeximport logging
import sys
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.llms.openai import OpenAI
!mkdir -p 'data/paul_graham/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'reader = SimpleDirectoryReader("./data/paul_graham/")
documents = reader.load_data()
data_generator = DatasetGenerator.from_documents(documents)
eval_questions = data_generator.generate_questions_from_nodes()
eval_questions
gpt4 = OpenAI(temperature=0, model="gpt-4")
evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)
vector_index = VectorStoreIndex.from_documents(documents)
def display_eval_df(query: str, response: Response, eval_result: str) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": (
                response.source_nodes[0].node.get_content()[:1000] + "..."
            ),
            "Evaluation Result": eval_result,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    display(eval_df)
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(eval_questions[1])
eval_result = evaluator_gpt4.evaluate_response(
    query=eval_questions[1], response=response_vector
)
display_eval_df(eval_questions[1], response_vector, eval_result)
