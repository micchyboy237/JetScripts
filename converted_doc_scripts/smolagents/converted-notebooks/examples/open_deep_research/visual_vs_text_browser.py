from browser_use import Agent
from dotenv import load_dotenv
from huggingface_hub import login
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from jet.libs.smolagents.examples.open_deep_research.scripts.gaia_scorer import (
    question_scorer,
)
from jet.libs.smolagents.examples.open_deep_research.scripts.run_agents import (
    answer_questions,
)
from jet.libs.smolagents.examples.open_deep_research.scripts.text_inspector_tool import (
    TextInspectorTool,
)
from jet.libs.smolagents.examples.open_deep_research.scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    NavigationalSearchTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from jet.libs.smolagents.examples.open_deep_research.scripts.visual_qa import (
    VisualQAGPT4Tool,
)
from smolagents import CodeAgent, LiteLLMModel
from smolagents import CodeAgent, LiteLLMModel, WebSearchTool
from smolagents.vision_web_browser import (
    close_popups,
    go_back,
    helium_instructions,
    initialize_agent,
    save_screenshot,
    search_item_ctrl_f,
)
import asyncio
import datasets
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Compare a text-based vs a vision-based browser

Warning: this notebook is experimental, it probably won't work out of the box!
"""
logger.info("# Compare a text-based vs a vision-based browser")

# !pip install "smolagents[litellm,toolkit]" -q


eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")["validation"]

to_keep = [
    "What's the last line of the rhyme under the flavor",
    'Of the authors (First M. Last) that worked on the paper "Pie Menus or Linear Menus',
    "In Series 9, Episode 11 of Doctor Who, the Doctor is trapped inside an ever-shifting maze. What is this location called in the official script for the episode? Give the setting exactly as it appears in the first scene heading.",
    "Which contributor to the version of OpenCV where support was added for the Mask-RCNN model has the same name as a former Chinese head of government when the names are transliterated to the Latin alphabet?",
    "The photograph in the Whitney Museum of American Art's collection with accession number 2022.128 shows a person holding a book. Which military unit did the author of this book join in 1813? Answer without using articles.",
    "I went to Virtue restaurant & bar in Chicago for my birthday on March 22, 2021 and the main course I had was delicious! Unfortunately, when I went back about a month later on April 21, it was no longer on the dinner menu.",
    "In Emily Midkiff's June 2014 article in a journal named for the one of Hreidmar's ",
    "Under DDC 633 on Bielefeld University Library's BASE, as of 2020",
    "In the 2018 VSCode blog post on replit.com, what was the command they clicked on in the last video to remove extra lines?",
    "The Metropolitan Museum of Art has a portrait in its collection with an accession number of 29.100.5. Of the consecrators and co-consecrators",
    "In Nature journal's Scientific Reports conference proceedings from 2012, in the article that did not mention plasmons or plasmonics, what nano-compound is studied?",
    'In the year 2022, and before December, what does "R" stand for in the three core policies of the type of content',
    "Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?",
]
eval_ds = eval_ds.filter(lambda row: any([el in row["Question"] for el in to_keep]))
eval_ds = eval_ds.rename_columns(
    {"Question": "question", "Final answer": "true_answer", "Level": "task"}
)


load_dotenv(override=True)

login(os.getenv("HF_TOKEN"))

"""
### Text browser
"""
logger.info("### Text browser")


proprietary_model = LiteLLMModel(model_id="gpt-4o")

browser = SimpleTextBrowser(
    viewport_size=1024 * 12,
    downloads_folder=os.path.join(OUTPUT_DIR, "downloads"),
    # searxng_url="http://your-searx-instance:8080"  # if you have one
)

WEB_TOOLS = [
    SearchInformationTool(browser),
    NavigationalSearchTool(browser),
    VisitTool(browser),
    PageUpTool(browser),
    PageDownTool(browser),
    FinderTool(browser),
    FindNextTool(browser),
    ArchiveSearchTool(browser),
]


surfer_agent = CodeAgent(
    model=proprietary_model,
    tools=WEB_TOOLS,
    max_steps=20,
    verbosity_level=2,
)

results_text = answer_questions(
    eval_ds,
    surfer_agent,
    "code_gpt4o_27-01_text",
    reformulation_model=proprietary_model,
    output_folder="output_browsers",
    visual_inspection_tool=VisualQAGPT4Tool(),
    text_inspector_tool=TextInspectorTool(proprietary_model, 40000),
)

"""
### Vision browser
"""
logger.info("### Vision browser")

# !pip install helium -q


proprietary_model = LiteLLMModel(model_id="gpt-4o")
vision_browser_agent = initialize_agent(proprietary_model)

CodeAgent(
    tools=[WebSearchTool(), go_back, close_popups, search_item_ctrl_f],
    model=proprietary_model,
    additional_authorized_imports=["helium"],
    step_callbacks=[save_screenshot],
    max_steps=20,
    verbosity_level=2,
)

results_vision = answer_questions(
    eval_ds,
    vision_browser_agent,
    "code_gpt4o_27-01_vision",
    reformulation_model=proprietary_model,
    output_folder="output_browsers",
    visual_inspection_tool=VisualQAGPT4Tool(),
    text_inspector_tool=TextInspectorTool(proprietary_model, 40000),
    postprompt=helium_instructions
    + "Any web browser controls won't work on .pdf urls, rather use the tool 'inspect_file_as_text' to read them",
)

"""
### Browser-use browser
"""
logger.info("### Browser-use browser")

# !pip install browser-use lxml_html_clean -q
# !playwright install


# import nest_asyncio


# nest_asyncio.apply()


load_dotenv()


class BrowserUseAgent:
    logs = []

    def write_inner_memory_from_logs(self, summary_mode):
        return self.results

    def run(self, task, **kwargs):
        agent = Agent(
            task=task,
            llm=ChatOllama(model="llama3.2"),
        )
        self.results = asyncio.get_event_loop().run_until_complete(agent.run())
        return self.results.history[-1].result[0].extracted_content


browser_use_agent = BrowserUseAgent()

results_browseruse = answer_questions(
    eval_ds,
    browser_use_agent,
    "gpt-4o_27-01_browseruse",
    reformulation_model=proprietary_model,
    output_folder="output_browsers",
    visual_inspection_tool=VisualQAGPT4Tool(),
    text_inspector_tool=TextInspectorTool(proprietary_model, 40000),
    postprompt="",
    run_simple=True,
)

"""
### Get results
"""
logger.info("### Get results")


results_vision, results_text, results_browseruse = (
    pd.DataFrame(results_vision),
    pd.DataFrame(results_text),
    pd.DataFrame(results_browseruse),
)

results_vision["is_correct"] = results_vision.apply(
    lambda x: question_scorer(x["prediction"], x["true_answer"]), axis=1
)
results_text["is_correct"] = results_text.apply(
    lambda x: question_scorer(x["prediction"], x["true_answer"]), axis=1
)
results_browseruse["is_correct"] = results_browseruse.apply(
    lambda x: question_scorer(x["prediction"], x["true_answer"]), axis=1
)

results = pd.concat([results_vision, results_text, results_browseruse])
results.groupby("agent_name")["is_correct"].mean()

correct_vision_results = results_vision.loc[results_vision["is_correct"]]
correct_vision_results

false_text_results = results_text.loc[~results_text["is_correct"]]
false_text_results

logger.info("\n\n[DONE]", bright=True)
