from autogen.agentchat.contrib.agent_eval.agent_eval import generate_criteria, quantify_criteria
from autogen.agentchat.contrib.agent_eval.criterion import Criterion
from autogen.agentchat.contrib.agent_eval.task import Task
from jet.logger import CustomLogger
from pathlib import Path
import autogen
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Demonstrating the `AgentEval` framework using the task of solving math problems as an example

This notebook aims to demonstrate how `AgentEval` works in an offline scenario, where we use a math problem-solving task as an example. 
`AgentEval` consists of two key steps:

- `generate_criteria`: This is an LLM-based function that generates a list of criteria $(c_1, \dots, c_n)$ to help to evaluate a utility given task.

- `quantify_criteria`: This function quantifies the performance of any sample task based on the criteria generated in the `generate_criteria` step in the following way: $(c_1=a_1, \dots, c_n=a_n)$

![AgentEval](https://media.githubusercontent.com/media/autogenhub/autogen/main/website/blog/2023-11-20-AgentEval/img/agenteval-CQ.png)

For more detailed explanations, please refer to the accompanying [blog post](https://autogenhub.github.io/autogen/blog/2023/11/20/AgentEval)

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install pyautogen, Docker, and Ollama:
"""
logger.info("# Demonstrating the `AgentEval` framework using the task of solving math problems as an example")

# %pip install "pyautogen>=0.2.3" docker
# %pip install scipy
# %pip install matplotlib

"""
## Set your API Endpoint
* The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file. It first looks for an environment variable with a specified name. The value of the environment variable needs to be a valid json string. If that variable is not found, it looks for a json file with the same name. It filters the configs by filter_dict.

You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/microsoft/autogen/blob/main/notebook/oai_openai_utils.ipynb) for full code examples of the different methods.
"""
logger.info("## Set your API Endpoint")




config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

"""
# Run the Critic

To run the critic, we need a couple of math problem examples. One of them failed to solve the problem successfully, given in `agenteval-in-out/response_failed.txt`, and the other one was solved successfully, i.e., `agenteval-in-out/response_successful.txt`.
"""
logger.info("# Run the Critic")

def remove_ground_truth(test_case):
    test_details = json.loads(test_case)
    correctness = test_details.pop("is_correct", None)
    test_details.pop("correct_ans", None)
    test_details.pop("check_result", None)
    return str(test_details), correctness


success_str = open("../test/test_files/agenteval-in-out/samples/sample_math_response_successful.txt", "r").read()
response_successful = remove_ground_truth(success_str)[0]
failed_str = open("../test/test_files/agenteval-in-out/samples/sample_math_response_failed.txt", "r").read()
response_failed = remove_ground_truth(failed_str)[0]

task = Task(
    **{
        "name": "Math problem solving",
        "description": "Given any question, the system needs to solve the problem as consisely and accurately as possible",
        "successful_response": response_successful,
        "failed_response": response_failed,
    }
)

criteria = generate_criteria(task=task, llm_config={"config_list": config_list}, max_round=8)

"""
# The Criteria
Now, we print the designed criteria for assessing math problems.
"""
logger.info("# The Criteria")

current_task_name = "_".join(task.name.split()).lower()
cr_file = open(f"../test/test_files/agenteval-in-out/{current_task_name}_criteria.json", "w")
cr_file.write(Criterion.write_json(criteria))
cr_file.close()

"""
*Note :* You can also define and use your own criteria in order to feed into the quantifier.

# The `QuantifierAgent`

Once we have the criteria, we need to quantify a new sample based on the designed criteria and its accepted values. This will be done through `quantify_criteria` from agent_eval. 
Again, you can use your own defined criteria in `criteria_file`.
"""
logger.info("# The `QuantifierAgent`")

criteria_file = f"../test/test_files/agenteval-in-out/{current_task_name}_criteria.json"
criteria = open(criteria_file, "r").read()
criteria = Criterion.parse_json_str(criteria)

"""
## Running the quantifier on a single test case

Here, we run the quantifier on a single math problem test case, `sample_test_case.json`, for demonstration.
"""
logger.info("## Running the quantifier on a single test case")

test_case = open("../test/test_files/agenteval-in-out/samples/sample_test_case.json", "r").read()
test_case, ground_truth = remove_ground_truth(test_case)
quantifier_output = quantify_criteria(
    llm_config={"config_list": config_list},
    criteria=criteria,
    task=task,
    test_case=test_case,
    ground_truth=ground_truth,
)
logger.debug("actual correctness:", quantifier_output["actual_success"])
logger.debug("predicted correctness:\n", quantifier_output["estimated_performance"])

"""
# Run `AgentEval` on the logs

In the example below, log_path points to the sample logs folder to run the quantifier. The current sample belongs to the prealgebra category which will be downloaded from [here](https://github.com/julianakiseleva/autogen/tree/agenteval/test/test_files/agenteval-in-out/samples).
In case you want to replicate the results described in the blog post, you can download all the logs for math problems using the following [link](https://github.com/julianakiseleva/autogen/tree/agenteval/model-logs/math-problems/agentchat).
"""
logger.info("# Run `AgentEval` on the logs")

log_path = "../test/test_files/agenteval-in-out/agentchat_results/"

# !wget https://github.com/julianakiseleva/autogen/raw/ddabd4f0e7c13a50e33cf8462e79358666371477/test/test_files/agenteval-in-out/prealgebra.zip
# !unzip -o prealgebra.zip -d {log_path}
# !rm  prealgebra.zip

assert Path(log_path).exists(), f"The log path '{log_path}' does not exist."

criteria_file = "../test/test_files/agenteval-in-out/samples/sample_math_criteria.json"
criteria = Criterion.parse_json_str(open(criteria_file, "r").read())
outcome = {}

for prefix in os.listdir(log_path):
    for file_name in os.listdir(log_path + "/" + prefix):
        gameid = prefix + "_" + file_name
        if file_name.split(".")[-1] == "json":
            test_case, ground_truth = remove_ground_truth(open(log_path + "/" + prefix + "/" + file_name, "r").read())
            quantifier_output = quantify_criteria(
                llm_config={"config_list": config_list},
                criteria=criteria,
                task=task,
                test_case=test_case,
                ground_truth=ground_truth,
            )
            outcome[gameid] = quantifier_output

with open("../test/test_files/agenteval-in-out/evaluated_problems.json", "w") as file:
    json.dump(outcome, file, indent=2)  # use `json.loads` to do the reverse

"""
## Plotting the estimated performance

Here you can find an example of how to visualize the obtained result in the histogram form (similar to the one in the blog post).
"""
logger.info("## Plotting the estimated performance")

try:
    criteria = Criterion.parse_json_str(open(criteria_file, "r").read())
except:  # noqa: E722
    pass


nl2int = {}
for criterion in criteria:
    score = 0
    for v in criterion.accepted_values:
        nl2int[v] = score
        score += 1
logger.debug(nl2int)

average_s = {}
average_f = {}

conf_interval_s = {}
conf_interval_f = {}

for criterion in criteria:
    task = {"s": [], "f": []}

    for game in outcome:
        try:
            tmp_dic = eval(outcome[game]["estimated_performance"])
            if outcome[game]["actual_success"] == "false":
                task["f"].append(nl2int[tmp_dic[criterion.name]])
            else:
                task["s"].append(nl2int[tmp_dic[criterion.name]])
        except:  # noqa: E722
            pass

    average_f[criterion.name] = np.mean(task["f"])
    average_s[criterion.name] = np.mean(task["s"])

    conf_interval_s[criterion.name] = stats.norm.interval(0.95, loc=np.mean(task["s"]), scale=stats.sem(task["s"]))
    conf_interval_f[criterion.name] = stats.norm.interval(0.95, loc=np.mean(task["f"]), scale=stats.sem(task["f"]))

"""
The final plot would be saved in `../test/test_files/agenteval-in-out/estimated_performance.png`
"""
logger.info("The final plot would be saved in `../test/test_files/agenteval-in-out/estimated_performance.png`")

plt.figure(figsize=(12, 8))
bar_width = 0.1
index = np.arange(len(criteria))


plt.bar(
    index,
    list(average_s.values()),
    bar_width,
    label=f"success ({len(task['s'])} samples)",
    color="darkblue",
    yerr=[(avg - conf_interval_s[key][0]) for key, avg in average_s.items()],
    capsize=5,
)
plt.bar(
    index + bar_width,
    list(average_f.values()),
    bar_width,
    label=f"failed ({len(task['f'])} samples)",
    color="lightblue",
    yerr=[(avg - conf_interval_f[key][0]) for key, avg in average_f.items()],
    capsize=5,
)

plt.xlabel("Criteria", fontsize=16)
plt.ylabel("Average Value", fontsize=16)
plt.title(
    "Average Values of 3 different baselines cases with 95% Confidence Intervals - math problems ", fontsize=12, pad=10
)  # Adjust titlepad to move the title further above
plt.xticks(index + bar_width / 2, [crit.name for crit in criteria], rotation=45, fontsize=14)
plt.legend(loc="upper center", fontsize=14, bbox_to_anchor=(0.5, 1), ncol=3)  # Adjust legend placement and ncol
plt.tight_layout()  # Adjust subplot parameters to fit the labels
plt.ylim(0, 5)
plt.savefig("../test/test_files/agenteval-in-out/estimated_performance.png")
plt.show()

logger.info("\n\n[DONE]", bright=True)