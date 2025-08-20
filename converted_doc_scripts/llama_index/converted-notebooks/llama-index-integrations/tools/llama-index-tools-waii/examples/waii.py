import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index import VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.tools.waii import WaiiToolSpec
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, lag, lead, round
from pyspark.sql.window import Window
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


waii_tool = WaiiToolSpec(
    url="https://tweakit.waii.ai/api/",
    api_key="3........",
    database_key="snowflake://....",
    verbose=True,
)


documents = waii_tool.load_data("Get all tables with their number of columns")
index = VectorStoreIndex.from_documents(documents).as_query_engine()

index.query(
    "Which table contains most columns, tell me top 5 tables with number of columns?"
).response


agent = FunctionAgent(
    waii_tool.to_tool_list(), llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)


ctx = Context(agent)

async def run_async_code_f23f048a():
    logger.debug(await agent.run("Give me top 3 countries with the most number of car factory", ctx=ctx))
    return 
 = asyncio.run(run_async_code_f23f048a())
logger.success(format_json())
async def run_async_code_5a1279a4():
    logger.debug(await agent.run("What are the car factories of these countries", ctx=ctx))
    return 
 = asyncio.run(run_async_code_5a1279a4())
logger.success(format_json())

logger.debug(
    await agent.run(
        "Give me top 3 longest running queries, include the complete query_id and their duration. And analyze performance of the first query",
        ctx=ctx,
    )
)

previous_query = """
SELECT
    employee_id,
    department,
    salary,
    AVG(salary) OVER (PARTITION BY department) AS department_avg_salary,
    salary - AVG(salary) OVER (PARTITION BY department) AS diff_from_avg
FROM
    employees;
"""
current_query = """
SELECT
    employee_id,
    department,
    salary,
    MAX(salary) OVER (PARTITION BY department) AS department_max_salary,
    salary - AVG(salary) OVER (PARTITION BY department) AS diff_from_avg
FROM
    employees;
LIMIT 100;
"""
async def run_async_code_27ab7a7a():
    logger.debug(await agent.run(f"tell me difference between {previous_query} and {current_query}", ctx=ctx))
    return 
 = asyncio.run(run_async_code_27ab7a7a())
logger.success(format_json())

async def run_async_code_6aabe5de():
    logger.debug(await agent.run("Summarize the dataset", ctx=ctx))
    return 
 = asyncio.run(run_async_code_6aabe5de())
logger.success(format_json())

q = """

spark = SparkSession.builder.appName("yearly_car_analysis").getOrCreate()

yearly_avg_hp = cars_data.groupBy("year").agg(avg("horsepower").alias("avg_horsepower"))

windowSpec = Window.orderBy("year")

yearly_comparisons = yearly_avg_hp.select(
    "year",
    "avg_horsepower",
    lag("avg_horsepower").over(windowSpec).alias("prev_year_hp"),
    lead("avg_horsepower").over(windowSpec).alias("next_year_hp")
)

final_result = yearly_comparisons.select(
    "year",
    "avg_horsepower",
    round(
        (yearly_comparisons.avg_horsepower - yearly_comparisons.prev_year_hp) /
        yearly_comparisons.prev_year_hp * 100, 2
    ).alias("percentage_diff_prev_year"),
    round(
        (yearly_comparisons.next_year_hp - yearly_comparisons.avg_horsepower) /
        yearly_comparisons.avg_horsepower * 100, 2
    ).alias("percentage_diff_next_year")
).orderBy("year")

final_result.show()
"""
async def run_async_code_4e31cd9c():
    logger.debug(await agent.run(f"translate this pyspark query {q}, to Snowflake", ctx=ctx))
    return 
 = asyncio.run(run_async_code_4e31cd9c())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)