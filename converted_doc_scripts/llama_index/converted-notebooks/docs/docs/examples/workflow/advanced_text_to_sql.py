import asyncio
from jet.transformers.formatters import format_json
from IPython.display import display, HTML
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import PromptTemplate
from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatResponse
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Context,
    Event,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.utils.workflow import draw_all_possible_flows
from pathlib import Path
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
from sqlalchemy import text
from typing import Dict
from typing import List
import json
import os
import pandas as pd
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Workflows for Advanced Text-to-SQL

<a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/docs/examples/workflow/advanced_text_to_sql.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In this guide we show you how to setup a text-to-SQL workflow over your data with our [workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows) syntax.

This gives you flexibility to enhance text-to-SQL with additional techniques. We show these in the below sections: 
1. **Query-Time Table Retrieval**: Dynamically retrieve relevant tables in the text-to-SQL prompt.
2. **Query-Time Sample Row retrieval**: Embed/Index each row, and dynamically retrieve example rows for each table in the text-to-SQL prompt.

Our out-of-the box workflows include our `NLSQLTableQueryEngine` and `SQLTableRetrieverQueryEngine`. (if you want to check out our text-to-SQL guide using these modules, take a look [here](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo.html)). This guide implements an advanced version of those modules, giving you the utmost flexibility to apply this to your own setting.

**NOTE:** Any Text-to-SQL application should be aware that executing 
arbitrary SQL queries can be a security risk. It is recommended to
take precautions as needed, such as using restricted roles, read-only
databases, sandboxing, etc.

## Load and Ingest Data


### Load Data
We use the [WikiTableQuestions dataset](https://ppasupat.github.io/WikiTableQuestions/) (Pasupat and Liang 2015) as our test dataset.

We go through all the csv's in one folder, store each in a sqlite database (we will then build an object index over each table schema).
"""
logger.info("# Workflows for Advanced Text-to-SQL")

# %pip install llama-index-llms-ollama

# !wget "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip" -O data.zip
# !unzip data.zip


data_dir = Path("./WikiTableQuestions/csv/200-csv")
csv_files = sorted([f for f in data_dir.glob("*.csv")])
dfs = []
for csv_file in csv_files:
    logger.debug(f"processing file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    except Exception as e:
        logger.debug(f"Error parsing {csv_file}: {str(e)}")

"""
### Extract Table Name and Summary from each Table

Here we use qwen3-1.7b-4bit-mini to extract a table name (with underscores) and summary from each table with our Pydantic program.
"""
logger.info("### Extract Table Name and Summary from each Table")

tableinfo_dir = "WikiTableQuestions_TableInfo"
# !mkdir {tableinfo_dir}


class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )


prompt_str = """\
Give me a summary of the table with the following JSON format.

- The table name must be unique to the table and describe it while being concise.
- Do NOT output a generic table name (e.g. table, my_table).

Do NOT make the table name one of the following: {exclude_table_name_list}

Table:
{table_str}

Summary: """
prompt_tmpl = ChatPromptTemplate(
    message_templates=[ChatMessage.from_str(prompt_str, role="user")]
)

llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit")


def _get_tableinfo_with_index(idx: int) -> str:
    results_gen = Path(tableinfo_dir).glob(f"{idx}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.parse_file(path)
    else:
        raise ValueError(
            f"More than one file matching index: {list(results_gen)}"
        )


table_names = set()
table_infos = []
for idx, df in enumerate(dfs):
    table_info = _get_tableinfo_with_index(idx)
    if table_info:
        table_infos.append(table_info)
    else:
        while True:
            df_str = df.head(10).to_csv()
            table_info = llm.structured_predict(
                TableInfo,
                prompt_tmpl,
                table_str=df_str,
                exclude_table_name_list=str(list(table_names)),
            )
            table_name = table_info.table_name
            logger.debug(f"Processed table: {table_name}")
            if table_name not in table_names:
                table_names.add(table_name)
                break
            else:
                logger.debug(
                    f"Table name {table_name} already exists, trying again.")
                pass

        out_file = f"{tableinfo_dir}/{idx}_{table_name}.json"
        json.dump(table_info.dict(), open(out_file, "w"))
    table_infos.append(table_info)

"""
### Put Data in SQL Database

We use `sqlalchemy`, a popular SQL database toolkit, to load all the tables.
"""
logger.info("### Put Data in SQL Database")


def sanitize_column_name(col_name):
    return re.sub(r"\W+", "_", col_name)


def create_table_from_dataframe(
    df: pd.DataFrame, table_name: str, engine, metadata_obj
):
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    table = Table(table_name, metadata_obj, *columns)

    metadata_obj.create_all(engine)

    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()


engine = create_engine("sqlite:///wiki_table_questions.db")
metadata_obj = MetaData()
for idx, df in enumerate(dfs):
    tableinfo = _get_tableinfo_with_index(idx)
    logger.debug(f"Creating table: {tableinfo.table_name}")
    create_table_from_dataframe(df, tableinfo.table_name, engine, metadata_obj)

"""
## Advanced Capability 1: Text-to-SQL with Query-Time Table Retrieval.

We now show you how to setup an e2e text-to-SQL with table retrieval.

### Define Modules

Here we define the core modules.
1. Object index + retriever to store table schemas
2. SQLDatabase object to connect to the above tables + SQLRetriever.
3. Text-to-SQL Prompt
4. Response synthesis Prompt
5. LLM

Object index, retriever, SQLDatabase
"""
logger.info(
    "## Advanced Capability 1: Text-to-SQL with Query-Time Table Retrieval.")


sql_database = SQLDatabase(engine)

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
    for t in table_infos
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

"""
SQLRetriever + Table Parser
"""
logger.info("SQLRetriever + Table Parser")


sql_retriever = SQLRetriever(sql_database)


def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


"""
Text-to-SQL Prompt + Output Parser
"""
logger.info("Text-to-SQL Prompt + Output Parser")


def parse_response_to_sql(chat_response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = chat_response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:"):]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
    dialect=engine.dialect.name
)
logger.debug(text2sql_prompt.template)

"""
Response Synthesis Prompt
"""
logger.info("Response Synthesis Prompt")

response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(
    response_synthesis_prompt_str,
)

llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit")

"""
### Define Workflow

Now that the components are in place, let's define the full workflow!
"""
logger.info("### Define Workflow")


class TableRetrieveEvent(Event):
    """Result of running table retrieval."""

    table_context_str: str
    query: str


class TextToSQLEvent(Event):
    """Text-to-SQL event."""

    sql: str
    query: str


class TextToSQLWorkflow1(Workflow):
    """Text-to-SQL Workflow that does query-time table retrieval."""

    def __init__(
        self,
        obj_retriever,
        text2sql_prompt,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        *args,
        **kwargs
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self.obj_retriever = obj_retriever
        self.text2sql_prompt = text2sql_prompt
        self.sql_retriever = sql_retriever
        self.response_synthesis_prompt = response_synthesis_prompt
        self.llm = llm

    @step
    def retrieve_tables(
        self, ctx: Context, ev: StartEvent
    ) -> TableRetrieveEvent:
        """Retrieve tables."""
        table_schema_objs = self.obj_retriever.retrieve(ev.query)
        table_context_str = get_table_context_str(table_schema_objs)
        return TableRetrieveEvent(
            table_context_str=table_context_str, query=ev.query
        )

    @step
    def generate_sql(
        self, ctx: Context, ev: TableRetrieveEvent
    ) -> TextToSQLEvent:
        """Generate SQL statement."""
        fmt_messages = self.text2sql_prompt.format_messages(
            query_str=ev.query, schema=ev.table_context_str
        )
        chat_response = self.llm.chat(fmt_messages)
        sql = parse_response_to_sql(chat_response)
        return TextToSQLEvent(sql=sql, query=ev.query)

    @step
    def generate_response(self, ctx: Context, ev: TextToSQLEvent) -> StopEvent:
        """Run SQL retrieval and generate response."""
        retrieved_rows = self.sql_retriever.retrieve(ev.sql)
        fmt_messages = self.response_synthesis_prompt.format_messages(
            sql_query=ev.sql,
            context_str=str(retrieved_rows),
            query_str=ev.query,
        )
        chat_response = llm.chat(fmt_messages)
        return StopEvent(result=chat_response)


"""
### Visualize Workflow

A really nice property of workflows is that you can both visualize the execution graph as well as a trace of the most recent execution.
"""
logger.info("### Visualize Workflow")


draw_all_possible_flows(
    TextToSQLWorkflow1, filename="text_to_sql_table_retrieval.html"
)


with open("text_to_sql_table_retrieval.html", "r") as file:
    html_content = file.read()

display(HTML(html_content))

"""
### Run Some Queries! 

Now we're ready to run some queries across this entire workflow.
"""
logger.info("### Run Some Queries!")

workflow = TextToSQLWorkflow1(
    obj_retriever,
    text2sql_prompt,
    sql_retriever,
    response_synthesis_prompt,
    llm,
    verbose=True,
)


async def async_func_9():
    response = await workflow.run(
        query="What was the year that The Notorious B.I.G was signed to Bad Boy?"
    )
    return response
response = asyncio.run(async_func_9())
logger.success(format_json(response))
logger.debug(str(response))


async def async_func_14():
    response = await workflow.run(
        query="Who won best director in the 1972 academy awards"
    )
    return response
response = asyncio.run(async_func_14())
logger.success(format_json(response))
logger.debug(str(response))


async def run_async_code_66b4deaf():
    async def run_async_code_be06940b():
        response = await workflow.run(query="What was the term of Pasquale Preziosa?")
        return response
    response = asyncio.run(run_async_code_be06940b())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_66b4deaf())
logger.success(format_json(response))
logger.debug(str(response))

"""
## 2. Advanced Capability 2: Text-to-SQL with Query-Time Row Retrieval (along with Table Retrieval)

One problem in the previous example is that if the user asks a query that asks for "The Notorious BIG" but the artist is stored as "The Notorious B.I.G", then the generated SELECT statement will likely not return any matches.

We can alleviate this problem by fetching a small number of example rows per table. A naive option would be to just take the first k rows. Instead, we embed, index, and retrieve k relevant rows given the user query to give the text-to-SQL LLM the most contextually relevant information for SQL generation.

We now extend our workflow.

### Index Each Table

We embed/index the rows of each table, resulting in one index per table.
"""
logger.info(
    "## 2. Advanced Capability 2: Text-to-SQL with Query-Time Row Retrieval (along with Table Retrieval)")


def index_all_tables(
    sql_database: SQLDatabase, table_index_dir: str = "table_index_dir"
) -> Dict[str, VectorStoreIndex]:
    """Index all tables."""
    if not Path(table_index_dir).exists():
        os.makedirs(table_index_dir)

    vector_index_dict = {}
    engine = sql_database.engine
    for table_name in sql_database.get_usable_table_names():
        logger.debug(f"Indexing rows in table: {table_name}")
        if not os.path.exists(f"{table_index_dir}/{table_name}"):
            with engine.connect() as conn:
                cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                result = cursor.fetchall()
                row_tups = []
                for row in result:
                    row_tups.append(tuple(row))

            nodes = [TextNode(text=str(t)) for t in row_tups]

            index = VectorStoreIndex(nodes)

            index.set_index_id("vector_index")
            index.storage_context.persist(f"{table_index_dir}/{table_name}")
        else:
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{table_index_dir}/{table_name}"
            )
            index = load_index_from_storage(
                storage_context, index_id="vector_index"
            )
        vector_index_dict[table_name] = index

    return vector_index_dict


vector_index_dict = index_all_tables(sql_database)

"""
### Define Expanded Table Parsing

We expand the capability of our table parsing to not only return the relevant table schemas, but also return relevant rows per table schema.

It now takes in both `table_schema_objs` (output of table retriever), but also the original `query_str` which will then be used for vector retrieval of relevant rows.
"""
logger.info("### Define Expanded Table Parsing")


sql_retriever = SQLRetriever(sql_database)


def get_table_context_and_rows_str(
    query_str: str,
    table_schema_objs: List[SQLTableSchema],
    verbose: bool = False,
):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        vector_retriever = vector_index_dict[
            table_schema_obj.table_name
        ].as_retriever(similarity_top_k=2)
        relevant_nodes = vector_retriever.retrieve(query_str)
        if len(relevant_nodes) > 0:
            table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
            for node in relevant_nodes:
                table_row_context += str(node.get_content()) + "\n"
            table_info += table_row_context

        if verbose:
            logger.debug(f"> Table Info: {table_info}")

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


"""
### Define Expanded Workflow

We re-use the workflow in section 1, but with an upgraded SQL parsing step after text-to-SQL generation.

It is very easy to subclass and extend an existing workflow, and customizing existing steps to be more advanced. Here we define a new worfklow that overrides the existing `retrieve_tables` step in order to return the relevant rows.
"""
logger.info("### Define Expanded Workflow")


class TextToSQLWorkflow2(TextToSQLWorkflow1):
    """Text-to-SQL Workflow that does query-time row AND table retrieval."""

    @step
    def retrieve_tables(
        self, ctx: Context, ev: StartEvent
    ) -> TableRetrieveEvent:
        """Retrieve tables."""
        table_schema_objs = self.obj_retriever.retrieve(ev.query)
        table_context_str = get_table_context_and_rows_str(
            ev.query, table_schema_objs, verbose=self._verbose
        )
        return TableRetrieveEvent(
            table_context_str=table_context_str, query=ev.query
        )


"""
Since the overall sequence of steps is the same, the graph should look the same.
"""
logger.info(
    "Since the overall sequence of steps is the same, the graph should look the same.")


draw_all_possible_flows(
    TextToSQLWorkflow2, filename="text_to_sql_table_retrieval.html"
)

"""
### Run Some Queries

We can now ask about relevant entries even if it doesn't exactly match the entry in the database.
"""
logger.info("### Run Some Queries")

workflow2 = TextToSQLWorkflow2(
    obj_retriever,
    text2sql_prompt,
    sql_retriever,
    response_synthesis_prompt,
    llm,
    verbose=True,
)


async def async_func_9():
    response = await workflow2.run(
        query="What was the year that The Notorious BIG was signed to Bad Boy?"
    )
    return response
response = asyncio.run(async_func_9())
logger.success(format_json(response))
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)
