from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from jet.logger import logger
from matplotlib.legend_handler import HandlerTuple  # Added import
from tqdm import tqdm
import datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
import string
import warnings


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

# !pip install -e .. datasets sympy numpy matplotlib seaborn -q  # Install dev version of smolagents + some packages

DATE = "2024-12-26"

EVAL_DATASET = "smolagents/benchmark-v1"

ANSWERS_DATASET = "smolagents/answers"
PUSH_ANSWERS_DATASET_TO_HUB = True

RESULTS_DATASET = "smolagents/results"
PUSH_RESULTS_DATASET_TO_HUB = True

"""
## Constants and utilities/tools
"""
logger.info("## Constants and utilities/tools")




def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def is_float(element: any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def normalize_str(input_str, remove_punct=True) -> str:
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase
    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)
    Returns:
    - str, the normalized string
    """
    no_spaces = re.sub(r"\s", "", input_str)

    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


def extract_numbers(text: str) -> list[str]:
    """This pattern matches:
    - Optional negative sign
    - Numbers with optional comma thousand separators
    - Optional decimal points with decimal numbers
    """
    pattern = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

    return [el.replace(",", "") for el in re.findall(pattern, text)]


def get_question_score_gaia(
    model_answer: str,
    ground_truth: str,
) -> bool:
    """Scoring function used to score functions from the GAIA benchmark"""
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(str(model_answer))
        return normalized_answer == float(ground_truth)

    elif any(char in ground_truth for char in [",", ";"]):  # if gt is a list
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        if len(gt_elems) != len(ma_elems):  # check length is the same
            warnings.warn("Answer lists have different lengths, returning False.", UserWarning)
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):  # compare each element as float or str
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False) == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    else:  # if gt is a str
        return normalize_str(model_answer) == normalize_str(ground_truth)


def get_correct(row):
    if row["source"] == "MATH":  # Checks the last number in answer
        numbers_answer = extract_numbers(str(row["answer"]))
        if len(numbers_answer) == 0:
            return False
        return np.isclose(float(numbers_answer[-1]), float(row["true_answer"]), rtol=1e-5, atol=1e-7)
    else:
        return get_question_score_gaia(str(row["answer"]), str(row["true_answer"]))


def score_answers_subset(answers_dataset, answers_subset):
    try:
        logger.debug(answers_dataset, answers_subset)
        *model_id, action_type, task = answers_subset.split("__")
        model_id = "/".join(model_id)
        ds = datasets.load_dataset(answers_dataset, answers_subset, split="test")
        df = ds.to_pandas()
        df["correct"] = df.apply(get_correct, axis=1)
        assert df["correct"].notnull().sum() > 30, "Missing answers"
        acc = df["correct"].mean().item()
        result = df.loc[0, ["model_id", "agent_action_type", "source"]].to_dict()
        result["acc"] = acc
        return result
    except Exception as e:
        logger.debug(f"Error with {answers_subset}: {e}")
        return None


def score_answers(
    answers_subsets,
    answers_dataset=ANSWERS_DATASET,
    date=DATE,
    push_to_hub_dataset=RESULTS_DATASET if PUSH_RESULTS_DATASET_TO_HUB else None,
    set_default=True,
):
    """
    Score answers from the given dataset subsets.

    Parameters:
        answers_subsets: List of dataset subsets to score
        answers_dataset: Dataset containing the answers
        date: Date to use for the config name
        push_to_hub_dataset: Dataset ID to push results to, or None to skip pushing
        set_default: If True, sets this config as the default config in the Hugging Face Hub dataset.
                     This means when users load the dataset without specifying a config,
                     this version will be loaded by default.
    """
    if not answers_dataset:
        raise ValueError("Pass 'answers_dataset' to load the answers from it")
    date = date or datetime.date.today().isoformat()
    results = []
    with ThreadPoolExecutor(max_workers=16) as exe:
        futures = [
            exe.submit(score_answers_subset, answers_dataset, answers_subset) for answers_subset in answers_subsets
        ]
        for f in tqdm(as_completed(futures), total=len(answers_subsets), desc="Processing tasks"):
            result = f.result()
            if result:
                results.append(result)
    df = pd.DataFrame(results)

    if push_to_hub_dataset:
        ds = datasets.Dataset.from_pandas(df)
        config = date
        ds.push_to_hub(push_to_hub_dataset, config_name=config, commit_message=f"Upload {config} results")
    return df

"""
## Score answers
"""
logger.info("## Score answers")



answers_subsets = datasets.get_dataset_config_names(ANSWERS_DATASET)
logger.debug("Number of answers_subsets", len(answers_subsets))
logger.debug("Example of answers_subset", answers_subsets[0])

result_df = score_answers(answers_subsets)
result_df["acc"] = (result_df["acc"] * 100).round(2)
result_df.head()

pivot_df = result_df.pivot_table(
    index=["model_id", "source"],
    columns=["agent_action_type"],
    values="acc",
    fill_value=float("nan"),
).reset_index()

"""
### Display results
"""
logger.info("### Display results")

display(pivot_df)



models = pivot_df["model_id"].unique()
sources = pivot_df["source"].unique()

plt.style.use("seaborn-v0_8-white")
fig, ax = plt.subplots(figsize=(15, 6))

width = 0.15  # width of each bar
spacing = 0.02  # space between bars within a group
group_spacing = 0.2  # space between model groups

num_sources = len(sources)
total_width_per_group = (width + spacing) * num_sources * 2  # *2 for agent and vanilla
x = np.arange(len(models)) * (total_width_per_group + group_spacing)

for i, source in enumerate(sources):
    source_data = pivot_df[pivot_df["source"] == source]
    agent_scores = [
        source_data[source_data["model_id"] == model]["code"].values[0]
        if len(source_data[source_data["model_id"] == model]) > 0
        else np.nan
        for model in models
    ]
    vanilla_scores = [
        source_data[source_data["model_id"] == model]["vanilla"].values[0]
        if len(source_data[source_data["model_id"] == model]) > 0
        else np.nan
        for model in models
    ]

    pos = x + i * (width * 2 + spacing)

    agent_bars = ax.bar(pos, agent_scores, width, label=f"{source} (Agent)", alpha=0.8)
    vanilla_bars = ax.bar(
        pos + width * 0.6,
        vanilla_scores,
        width,
        hatch="////",
        alpha=0.5,
        hatch_linewidth=2,
        label=f"{source} (Vanilla)",
        color="white",
        edgecolor=agent_bars[0].get_facecolor(),
    )

ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")

group_centers = x + (total_width_per_group - spacing) / 2
ax.set_xticks(group_centers)

wrapped_labels = ["\n".join(model.split("/")) for model in models]
ax.set_xticklabels(wrapped_labels, rotation=0, ha="center")

handles, labels = ax.get_legend_handles_labels()
unique_sources = sources
legend_elements = [
    (handles[i * 2], handles[i * 2 + 1], labels[i * 2].replace(" (Agent)", "")) for i in range(len(unique_sources))
]
custom_legend = ax.legend(
    [(agent_handle, vanilla_handle) for agent_handle, vanilla_handle, _ in legend_elements],
    [label for _, _, label in legend_elements],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.show()

logger.info("\n\n[DONE]", bright=True)