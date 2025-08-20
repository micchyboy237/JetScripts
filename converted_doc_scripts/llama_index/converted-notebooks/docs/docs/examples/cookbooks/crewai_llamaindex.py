from crewai import Agent, Task, Crew, Process
from crewai_tools import LlamaIndexTool
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# CrewAI + LlamaIndex Cookbook

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/crewai_llamaindex.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This cookbook shows you how to build an advanced research assistant by plugging LlamaIndex-powered tools into a CrewAI-powered multi-agent setup.

LlamaIndex is a framework enabling developers to easily build LLM-powered applications over their data; it contains production modules for indexing, retrieval, and prompt/agent orchestration. A core use case is building a generalized QA interface enabling knowledge synthesis over complex questions.

Plugging a LlamaIndex RAG pipeline as a tool into a CrewAI agent setup enables even more sophisticated/advanced research flows as shown below. We show the following:

1. **Setup a Simple Calculator Agent**: We port over the set of tools available on LlamaHub (https://llamahub.ai/).
2. **Setup a Financial Analyst Agent**: We plug in a RAG query engine as a tool accessible to a CrewAI agent.
"""
logger.info("# CrewAI + LlamaIndex Cookbook")


# !pip install llama-index-core
# !pip install llama-index-readers-file
# !pip install llama-index-tools-wolfram-alpha
# !pip install 'crewai[tools]'

"""
## Setup a Simple Calculator Agent

In this section we setup a crew of agents that can perform math and generate a 10-question multiple choice test (with answers) from it.

#### Wolfram Alpha Tool
Let's setup Wolfram Alpha as a general math computation tool.
"""
logger.info("## Setup a Simple Calculator Agent")


wolfram_spec = WolframAlphaToolSpec(app_id="<app_id>")
wolfram_tools = wolfram_spec.to_tool_list()

wolfram_tools[0]("(7 * 12 ^ 10) / 321")

wolfram_tools[0]("How many calories are there in a pound of apples")

crewai_wolfram_tools = [LlamaIndexTool.from_tool(t) for t in wolfram_tools]

logger.debug(crewai_wolfram_tools[0].description)

calculator_agent = Agent(
    role="Calculator",
    goal="Solve complex math problems",
    backstory="""You are an AI computer that has access to Wolfram Alpha to perform complex computations.""",
    verbose=True,
    tools=crewai_wolfram_tools,
)
teacher_agent = Agent(
    role="Math Teacher",
    goal="Make tests for students.",
    backstory="""You are a math teacher preparing a simple arithmetic test for your 2nd grade students.""",
    verbose=True,
    allow_delegation=False,
)

task1 = Task(
    description="""Using the math operators (+, -, *, /), and numbers from 1-100, generate 10 medium-difficulty arithmetic problems
  that consist of numbers/operators/parentheses in different ways.

  Generate the actual answer for each problem too. Use the Wolfram tool for this.
  """,
    expected_output="10 arithmetic expressions with the actual answers",
    agent=calculator_agent,
)

task2 = Task(
    description="""Using the generated expressions/answers, generate a multiple choice for students.
  Each question should have 4 options, one being the correct answer. """,
    expected_output="Test with 10 multiple choice questions",
    agent=teacher_agent,
)

crew = Crew(
    agents=[calculator_agent, teacher_agent],
    tasks=[task1, task2],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

result = crew.kickoff()

logger.debug("######################")
logger.debug(result)

"""
## Setup a Simple Financial Analyst Agent

In this section we setup a crew that uses a LlamaIndex RAG pipeline over an Uber 10K as its core query tool.
"""
logger.info("## Setup a Simple Financial Analyst Agent")

# !wget "https://s23.q4cdn.com/407969754/files/doc_financials/2019/ar/Uber-Technologies-Inc-2019-Annual-Report.pdf" -O uber_10k.pdf



reader = SimpleDirectoryReader(input_files=["uber_10k.pdf"])
docs = reader.load_data()

docs[1].get_content()

llm = MLX(model="qwen3-1.7b-4bit")
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)

query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Uber 2019 10K Query Tool",
    description="Use this tool to lookup the 2019 Uber 10K Annual Report",
)

query_tool.args_schema.schema()

"""
## Generate a Research Report

Now that we have the query interface over the Uber 10K setup with LlamaIndex, we can now generate a research report with CrewAI.


We follow the agent/writer setup in the CrewAI quickstart tutorial, and modify it to use the query tool.

We then run it and analyze the results.
"""
logger.info("## Generate a Research Report")

researcher = Agent(
    role="Senior Financial Analyst",
    goal="Uncover insights about different tech companies",
    backstory="""You work at an asset management firm.
  Your goal is to understand tech stocks like Uber.""",
    verbose=True,
    allow_delegation=False,
    tools=[query_tool],
)
writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content on tech advancements",
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=False,
)

task1 = Task(
    description="""Conduct a comprehensive analysis of Uber's risk factors in 2019.""",
    expected_output="Full analysis report in bullet points",
    agent=researcher,
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog
  post that highlights the headwinds that Uber faces.
  Your post should be informative yet accessible, catering to a casual audience.
  Make it sound cool, avoid complex words.""",
    expected_output="Full blog post of at least 4 paragraphs",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

result = crew.kickoff()

logger.debug("######################")
logger.debug(result)

logger.info("\n\n[DONE]", bright=True)