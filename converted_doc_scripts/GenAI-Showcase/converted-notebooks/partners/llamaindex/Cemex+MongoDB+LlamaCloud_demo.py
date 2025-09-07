from google.colab import userdata
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
from jet.logger import CustomLogger
from llama_cloud_services import LlamaParse
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


"""
# Handling product knowledge Q&A with pre-processing

We were given [a spreadsheet](https://www.dropbox.com/scl/fi/4ls998569fgbo9zjn5tpv/matrix_expertos.xlsx?rlkey=ktd3hchpei60q4pm3o1c31lal&dl=0) containing a matrix of concrete products and appropriate applications for them.

The challenge is to answer natural-language questions about the products in the spreadsheet, which is very large and complex.

We take a multi-stage approach using a lengthy pre-processing step.

First install all our deps:
* MongoDB as our vector store
* LlamaParse for parsing the spreadsheet
* Ollama for understanding the spreadsheet
* Ollama embeddings to embed the data
"""
logger.info("# Handling product knowledge Q&A with pre-processing")

# !pip install llama-index-core llama-cloud llama-cloud-services llama-index-llms-anthropic llama-index-indices-managed-llama-cloud

"""
Connect to our llamacloud index:
"""
logger.info("Connect to our llamacloud index:")


index = LlamaCloudIndex(
    name="mongodb-cemex-demo",
    project_name="Rando project",
    organization_id="e793a802-cb91-4e6a-bd49-61d0ba2ac5f9",
    api_key=userdata.get("llamacloud-cemex"),
)

# import nest_asyncio

# nest_asyncio.apply()

"""
We parse the spreadsheet into a single markdown document, which LLMs find easier to understand:
"""
logger.info("We parse the spreadsheet into a single markdown document, which LLMs find easier to understand:")


parser = LlamaParse(result_type="markdown", api_key=userdata.get("llamacloud-cemex"))
documents = parser.load_data("data/matrix_expertos.xlsx")

"""
We get a very large Markdown table:
"""
logger.info("We get a very large Markdown table:")

raw_output = documents[0].text
logger.debug(raw_output)

"""
We confirm that Claude 3.7 Sonnet is capable of understanding what it's looking at by asking it to translate the headers:
"""
logger.info("We confirm that Claude 3.7 Sonnet is capable of understanding what it's looking at by asking it to translate the headers:")


llm = Ollama(
    model="claude-3-7-sonnet-20250219", api_key=userdata.get("anthropic-key")
)

response = llm.complete(
    f"You are looking at a spreadsheet in the form of a large markdown table. Translate the labels into english. <spreadsheet>{raw_output}</spreadsheet>"
)

logger.debug(response)

"""
We now use Claude to transform the data from spreadsheet rows and columns to a series of declarative statements, one for each property in the spreadsheet. This greatly expands the volume of data being processed but makes it much more amenable to semantic search, since the meaning of each "X" is expanded into its plain semantic meaning (in Spanish).

This involves several subtleties:
* We need to extend our `max_tokens` to allow the longest possible output from the model
* The output is still much longer than `max_tokens`, so we modify the prompt to focus on one section of the spreadsheet at a time (there are conveniently four top-level subsegments we can use for this purpose)
* Though the instructions are in English the requested output remains in Spanish since the expected questions are in that language
* Processing this step takes about 20 minutes
"""
logger.info("We now use Claude to transform the data from spreadsheet rows and columns to a series of declarative statements, one for each property in the spreadsheet. This greatly expands the volume of data being processed but makes it much more amenable to semantic search, since the meaning of each "X" is expanded into its plain semantic meaning (in Spanish).")

long_response_llm = llm = Ollama(
    model="claude-3-7-sonnet-20250219",
    api_key=userdata.get("anthropic-high-volume"),
    max_tokens=64000,
)

subsegments = ["Commercial", "Industrial", "Infrastructure", "Housing"]

all_responses = ""
for subsegment in subsegments:
    response = long_response_llm.complete(
        f"""
    You are given a large table derived from a spreadsheet.

    The first four columns define a possible place a product could be used, aka their applications. They are:
    - **Subsegmento** -> Subsegment (commercial, industrial, infrastructure, housing)
    - **Especialidad** -> Specialty (hotels, supermarkets, healthcare, industrial parks, etc.)
    - **Tipo de obra** -> Type of work (parking, hydraulic/sanitary work, exteriors, etc.)
    - **Elementos** -> Elements (such as slabs, beams, columns, walls, etc.)

    The remaining columns are products, solutions, and "multi-products".

    If the product, solution or multi-product is appropriate for the application, there is an X or an x in the corresponding column.

    Convert the table into a series of statements (in Spanish), e.g.
    "Biocrete Antihongo-Antialga es apropiado para bases hidráulicas/sanitarias en edificaciones verticales"

    Do this only for applications in the subsegment <subsegment>{subsegment}</subsegment>

    Every X in the spreadsheet should correspond to a statement.
    Include a response for every single X in this subsegment. DO NOT leave any out or abbreviate, even if the response is very long.
    Do not include any preamble or explanation, just give the list of statements.

    <spreadsheet>{raw_output}</spreadsheet>
  """
    )
    logger.debug(str(response))
    all_responses += str(response) + "\n"

logger.debug(all_responses)

"""
We get a set of thousands of statements, one for each X in the spreadsheet.
"""
logger.info("We get a set of thousands of statements, one for each X in the spreadsheet.")

logger.debug(len(str(all_responses).split("\n")))

"""
We now convert our combined statements into embeddings. This means we only need to do our conversion step one time, and thereafter the statements are stored in Mongo.
"""
logger.info("We now convert our combined statements into embeddings. This means we only need to do our conversion step one time, and thereafter the statements are stored in Mongo.")


document = Document(text=all_responses)

index.insert(document)

"""
We now create a query engine from the index. Since we only want concise answers, we can switch back to our LLM with much lower `max_tokens` value.
"""
logger.info("We now create a query engine from the index. Since we only want concise answers, we can switch back to our LLM with much lower `max_tokens` value.")


Settings.llm = llm  # this is the short one
query_engine = index.as_query_engine()

"""
We provide our sample query in Spanish and get back a Spanish-language answer:
"""
logger.info("We provide our sample query in Spanish and get back a Spanish-language answer:")

response = query_engine.query(
    "para la especialidad de comercio para la especial de hotelería qué concreto me recomiendas para muros de un edificio?"
)
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)