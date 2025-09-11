from collections import Counter
from dotenv import load_dotenv
from huggingface_hub import login
from jet.logger import logger
from scripts.gaia_scorer import check_close_call, question_scorer
import datasets
import glob
import numpy as np
import os
import pandas as pd
import plotly.express as px
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

# !pip install plotly kaleido datasets nbformat -U -q




load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

pd.set_option("max_colwidth", None)

OUTPUT_DIR = "output"

eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")["validation"]
eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})
eval_df = pd.DataFrame(eval_ds)

"""
# 1. Load all results
"""
logger.info("# 1. Load all results")



results = []
for f in glob.glob(f"{OUTPUT_DIR}/validation/*.jsonl"):
    df = pd.read_json(f, lines=True)
    df["agent_name"] = f.split("/")[-1].split(".")[0]
    results.append(df)

result_df = pd.concat(results)
result_df["prediction"] = result_df["prediction"].fillna("No prediction")




result_df["is_correct"] = result_df.apply(lambda x: question_scorer(x["prediction"], x["true_answer"]), axis=1)
result_df["is_near_correct"] = result_df.apply(
    lambda x: check_close_call(x["prediction"], x["true_answer"], x["is_correct"]),
    axis=1,
)

result_df["count_steps"] = result_df["intermediate_steps"].apply(len)


def find_attachment(question):
    matches = eval_df.loc[eval_df["question"].apply(lambda x: x in question), "file_name"]

    if len(matches) == 0:
        return "Not found"
    file_path = matches.values[0]

    if isinstance(file_path, str) and len(file_path) > 0:
        return file_path.split(".")[-1]
    else:
        return "None"


result_df["attachment_type"] = result_df["question"].apply(find_attachment)


def extract_tool_calls(code):
    regex = r"\b(\w+)\("
    function_calls = [el for el in re.findall(regex, code) if el.islower()]

    function_call_counter = Counter(function_calls)
    return function_call_counter


def sum_tool_calls(steps):
    total_count = Counter()
    for step in steps:
        if "llm_output" in step:
            total_count += extract_tool_calls(step["llm_output"])

    return total_count


def get_durations(row):

    duration_timedelta = row["end_time"] - row["start_time"]
    return int(duration_timedelta.total_seconds())


result_df["duration"] = result_df.apply(get_durations, axis=1)

result_df["agent_name"].value_counts()

"""
# 2. Inspect specific runs
"""
logger.info("# 2. Inspect specific runs")

sel_df = result_df
sel_df = sel_df.reset_index(drop=True)
display(sel_df["agent_name"].value_counts())
sel_df = sel_df.drop_duplicates(subset=["agent_name", "question"])
display(sel_df.groupby("agent_name")[["task"]].value_counts())
logger.debug("Total length:", len(sel_df), "- is complete:", len(sel_df) == 165)

display("Average score:", sel_df.groupby("agent_name")[["is_correct"]].mean().round(3))
display(
    sel_df.groupby(["agent_name", "task"])[["is_correct", "is_near_correct", "count_steps", "question", "duration"]]
    .agg(
        {
            "is_correct": "mean",
            "is_near_correct": "mean",
            "count_steps": "mean",
            "question": "count",
            "duration": "mean",
        }
    )
    .rename(columns={"question": "count"})
)



cumulative_df = (
    (
        sel_df.groupby("agent_name")[["is_correct", "is_near_correct"]]
        .expanding(min_periods=1, axis=0, method="single")
        .agg({"is_correct": "mean", "is_near_correct": "count"})
        .reset_index()
    )
    .copy()
    .rename(columns={"is_near_correct": "index"})
)
cumulative_df["index"] = cumulative_df["index"].astype(int) - 1


def find_question(row):
    try:
        res = sel_df.loc[sel_df["agent_name"] == row["agent_name"], "question"].iloc[row["index"]][:50]
        return res
    except Exception:
        return ""


cumulative_df["question"] = cumulative_df.apply(find_question, axis=1)

px.line(
    cumulative_df,
    color="agent_name",
    x="index",
    y="is_correct",
    hover_data="question",
)

"""
# 3. Dive deeper into one run
"""
logger.info("# 3. Dive deeper into one run")

sel_df = result_df.loc[result_df["agent_name"] == "o1"]
logger.debug(len(sel_df))

"""
### Count errors
"""
logger.info("### Count errors")



error_types = [
    "AgentParsingError",
    "AgentExecutionError",
    "AgentMaxIterationsError",
    "AgentGenerationError",
]
sel_df[error_types] = 0
sel_df["Count steps"] = np.nan


def count_errors(row):
    if isinstance(row["intermediate_steps"], list):
        row["Count steps"] = len(row["intermediate_steps"])
        for step in row["intermediate_steps"]:
            if isinstance(step, dict) and "error" in step:
                try:
                    row[str(step["error"]["error_type"])] += 1
                except Exception:
                    pass
    return row


sel_df = sel_df.apply(count_errors, axis=1)



aggregate_errors = (
    sel_df.groupby(["is_correct"])[error_types + ["Count steps"]].mean().reset_index().melt(id_vars=["is_correct"])
)

fig = px.bar(
    aggregate_errors,
    y="value",
    x="variable",
    color="is_correct",
    labels={
        "agent_name": "<b>Model</b>",
        "task": "<b>Level</b>",
        "aggregate_score": "<b>Performance</b>",
        "value": "<b>Average count</b>",
        "eval_score_GPT4": "<b>Score</b>",
    },
)
fig.update_layout(
    height=500,
    width=800,
    barmode="group",
    bargroupgap=0.0,
)
fig.update_traces(textposition="outside")
fig.write_image("aggregate_errors.png", scale=3)
fig.show()

"""
### Inspect result by file extension type
"""
logger.info("### Inspect result by file extension type")

display(
    result_df.groupby(["attachment_type"])[["is_correct", "count_steps", "question"]].agg(
        {"is_correct": "mean", "count_steps": "mean", "question": "count"}
    )
)

"""
# 4. Ensembling methods
"""
logger.info("# 4. Ensembling methods")

counts = result_df["agent_name"].value_counts()
long_series = result_df.loc[result_df["agent_name"].isin(counts[counts > 140].index)]

def majority_vote(df):
    df = df[(df["prediction"] != "Unable to determine") & (~df["prediction"].isna()) & (df["prediction"] != "None")]

    answer_modes = df.groupby("question")["prediction"].agg(lambda x: x.mode()[0]).reset_index()
    first_occurrences = (
        df.groupby(["question", "prediction"]).agg({"task": "first", "is_correct": "first"}).reset_index()
    )
    result = answer_modes.merge(first_occurrences, on=["question", "prediction"], how="left")

    return result


def oracle(df):
    def get_first_correct_or_first_wrong(group):
        correct_answers = group[group["is_correct"]]
        if len(correct_answers) > 0:
            return correct_answers.iloc[0]
        return group.iloc[0]

    result = df.groupby("question").apply(get_first_correct_or_first_wrong)

    return result.reset_index(drop=True)


display((long_series.groupby("agent_name")["is_correct"].mean() * 100).round(2))
logger.debug(f"Majority score: {majority_vote(long_series)['is_correct'].mean() * 100:.2f}")
logger.debug(f"Oracle score: {oracle(long_series)['is_correct'].mean() * 100:.2f}")

"""
### Submit
"""
logger.info("### Submit")

agent_run = "code_o1_04_february_submission5.jsonl"
df = pd.read_json(f"output/validation/{agent_run}", lines=True)
df = df[["task_id", "prediction", "intermediate_steps"]]
df = df.rename(columns={"prediction": "model_answer", "intermediate_steps": "reasoning_trace"})

df.to_json("submission.jsonl", orient="records", lines=True)

logger.info("\n\n[DONE]", bright=True)