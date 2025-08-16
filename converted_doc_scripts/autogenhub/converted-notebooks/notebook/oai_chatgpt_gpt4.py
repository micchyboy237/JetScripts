from autogen.math_utils import eval_math_responses
from jet.logger import CustomLogger
import autogen
import datasets
import logging
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


"""
<a href="https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/oai_chatgpt_gpt4.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Contributions to this project, i.e., https://github.com/autogenhub/autogen, are licensed under the Apache License, Version 2.0 (Apache-2.0).
Copyright (c) 2023 - 2024, Owners of https://github.com/autogenhub
SPDX-License-Identifier: Apache-2.0
Portions derived from  https://github.com/microsoft/autogen under the MIT License.
SPDX-License-Identifier: MIT
Copyright (c) Microsoft Corporation. All rights reserved. 

Licensed under the MIT License.

# Use AutoGen to Tune ChatGPT

AutoGen offers a cost-effective hyperparameter optimization technique [EcoOptiGen](https://arxiv.org/abs/2303.04673) for tuning Large Language Models. The study finds that tuning hyperparameters can significantly improve the utility of LLMs.
Please find documentation about this feature [here](/docs/Use-Cases/AutoGen#enhanced-inference).

In this notebook, we tune Ollama ChatGPT (both GPT-3.5 and GPT-4) models for math problem solving. We use [the MATH benchmark](https://crfm.stanford.edu/helm/latest/?group=math_chain_of_thought) for measuring mathematical problem solving on competition math problems with chain-of-thoughts style reasoning.

Related link: [Blogpost](https://autogenhub.github.io/autogen/blog/2023/04/21/LLM-tuning-math) based on this experiment.

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install with the [blendsearch] option:
```bash
pip install "autogen[blendsearch]"
```
"""
logger.info("# Use AutoGen to Tune ChatGPT")



"""
AutoGen has provided an API for hyperparameter optimization of Ollama ChatGPT models: `autogen.ChatCompletion.tune` and to make a request with the tuned config: `autogen.ChatCompletion.create`. First, we import autogen:
"""
logger.info("AutoGen has provided an API for hyperparameter optimization of Ollama ChatGPT models: `autogen.ChatCompletion.tune` and to make a request with the tuned config: `autogen.ChatCompletion.create`. First, we import autogen:")




"""
### Set your API Endpoint

The [`config_list_openai_aoai`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_openai_aoai) function tries to create a list of  Azure Ollama endpoints and Ollama endpoints. It assumes the api keys and api bases are stored in the corresponding environment variables or local txt files:

# - Ollama API key: os.environ["OPENAI_API_KEY"] or `openai_api_key_file="key_openai.txt"`.
# - Azure Ollama API key: os.environ["AZURE_OPENAI_API_KEY"] or `aoai_api_key_file="key_aoai.txt"`. Multiple keys can be stored, one per line.
- Azure Ollama API base: os.environ["AZURE_OPENAI_API_BASE"] or `aoai_api_base_file="base_aoai.txt"`. Multiple bases can be stored, one per line.

It's OK to have only the Ollama API key, or only the Azure Ollama API key + base.
"""
logger.info("### Set your API Endpoint")

config_list = autogen.config_list_openai_aoai()

"""
The config list looks like the following:
```python
config_list = [
    {'api_key': '<your Ollama API key here>'},  # only if Ollama API key is found
    {
        'api_key': '<your first Azure Ollama API key here>',
        'base_url': '<your first Azure Ollama API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },  # only if at least one Azure Ollama API key is found
    {
        'api_key': '<your second Azure Ollama API key here>',
        'base_url': '<your second Azure Ollama API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },  # only if the second Azure Ollama API key is found
]
```

You can directly override it if the above function returns an empty list, i.e., it doesn't find the keys in the specified locations.

## Load dataset

We load the competition_math dataset. The dataset contains 201 "Level 2" Algebra examples. We use a random sample of 20 examples for tuning the generation hyperparameters and the remaining for evaluation.
"""
logger.info("## Load dataset")

seed = 41
data = datasets.load_dataset("competition_math")
train_data = data["train"].shuffle(seed=seed)
test_data = data["test"].shuffle(seed=seed)
n_tune_data = 20
tune_data = [
    {
        "problem": train_data[x]["problem"],
        "solution": train_data[x]["solution"],
    }
    for x in range(len(train_data))
    if train_data[x]["level"] == "Level 2" and train_data[x]["type"] == "Algebra"
][:n_tune_data]
test_data = [
    {
        "problem": test_data[x]["problem"],
        "solution": test_data[x]["solution"],
    }
    for x in range(len(test_data))
    if test_data[x]["level"] == "Level 2" and test_data[x]["type"] == "Algebra"
]
logger.debug(len(tune_data), len(test_data))

"""
Check a tuning example:
"""
logger.info("Check a tuning example:")

logger.debug(tune_data[1]["problem"])

"""
Here is one example of the canonical solution:
"""
logger.info("Here is one example of the canonical solution:")

logger.debug(tune_data[1]["solution"])

"""
## Define Success Metric

Before we start tuning, we must define the success metric we want to optimize. For each math task, we use voting to select a response with the most common answers out of all the generated responses. We consider the task successfully solved if it has an equivalent answer to the canonical solution. Then we can optimize the mean success rate of a collection of tasks.

## Use the tuning data to find a good configuration

For (local) reproducibility and cost efficiency, we cache responses from Ollama with a controllable seed.
"""
logger.info("## Define Success Metric")

autogen.ChatCompletion.set_cache(seed)

"""
This will create a disk cache in ".cache/{seed}". You can change `cache_path_root` from ".cache" to a different path in `set_cache()`. The cache for different seeds are stored separately.

### Perform tuning

The tuning will take a while to finish, depending on the optimization budget. The tuning will be performed under the specified optimization budgets.

* `inference_budget` is the benchmark's target average inference budget per instance. For example, 0.004 means the target inference budget is 0.004 dollars, which translates to 2000 tokens (input + output combined) if the gpt-3.5-turbo model is used.
* `optimization_budget` is the total budget allowed for tuning. For example, 1 means 1 dollar is allowed in total, which translates to 500K tokens for the gpt-3.5-turbo model.
* `num_sumples` is the number of different hyperparameter configurations allowed to be tried. The tuning will stop after either num_samples trials are completed or optimization_budget dollars are spent, whichever happens first. -1 means no hard restriction in the number of trials and the actual number is decided by `optimization_budget`.

Users can specify tuning data, optimization metric, optimization mode, evaluation function, search spaces etc.. The default search space is:

```python
default_search_space = {
    "model": tune.choice([
        "gpt-3.5-turbo",
        "gpt-4",
    ]),
    "temperature_or_top_p": tune.choice(
        [
            {"temperature": tune.uniform(0, 2)},
            {"top_p": tune.uniform(0, 1)},
        ]
    ),
    "max_tokens": tune.lograndint(50, 1000),
    "n": tune.randint(1, 100),
    "prompt": "{prompt}",
}
```

The default search space can be overridden by users' input.
For example, the following code specifies a fixed prompt template. The default search space will be used for hyperparameters that don't appear in users' input.
"""
logger.info("### Perform tuning")

prompts = [
    "{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{{}}."
]
config, analysis = autogen.ChatCompletion.tune(
    data=tune_data,  # the data for tuning
    metric="success_vote",  # the metric to optimize
    mode="max",  # the optimization mode
    eval_func=eval_math_responses,  # the evaluation function to return the success metrics
    inference_budget=0.02,  # the inference budget (dollar per instance)
    optimization_budget=1,  # the optimization budget (dollar in total)
    num_samples=20,
    model="llama3.2", request_timeout=300.0, context_window=4096,  # comment to tune both gpt-3.5-turbo and gpt-4
    prompt=prompts,  # the prompt templates to choose from
    config_list=config_list,  # the endpoint list
    allow_format_str_template=True,  # whether to allow format string template
)

"""
### Output tuning results

After the tuning, we can print out the config and the result found by AutoGen, which uses flaml for tuning.
"""
logger.info("### Output tuning results")

logger.debug("optimized config", config)
logger.debug("best result on tuning data", analysis.best_result)

"""
### Make a request with the tuned config

We can apply the tuned config on the request for an example task:
"""
logger.info("### Make a request with the tuned config")

response = autogen.ChatCompletion.create(context=tune_data[1], config_list=config_list, **config)
metric_results = eval_math_responses(autogen.ChatCompletion.extract_text(response), **tune_data[1])
logger.debug("response on an example data instance:", response)
logger.debug("metric_results on the example data instance:", metric_results)

"""
### Evaluate the success rate on the test data

You can use `autogen.ChatCompletion.test` to evaluate the performance of an entire dataset with the tuned config. The following code will take a while (30 mins to 1 hour) to evaluate all the test data instances if uncommented and run. It will cost roughly $3.
"""
logger.info("### Evaluate the success rate on the test data")



"""
What about the default, untuned gpt-4 config (with the same prompt as the tuned config)? We can evaluate it and compare:
"""
logger.info("What about the default, untuned gpt-4 config (with the same prompt as the tuned config)? We can evaluate it and compare:")



"""
The default use of GPT-4 has a much lower accuracy. Note that the default config has a lower inference cost. What if we heuristically increase the number of responses n?
"""
logger.info("The default use of GPT-4 has a much lower accuracy. Note that the default config has a lower inference cost. What if we heuristically increase the number of responses n?")



"""
The inference cost is doubled and matches the tuned config. But the success rate doesn't improve much. What if we further increase the number of responses n to 5?
"""
logger.info("The inference cost is doubled and matches the tuned config. But the success rate doesn't improve much. What if we further increase the number of responses n to 5?")



"""
We find that the 'success_vote' metric is increased at the cost of exceeding the inference budget. But the tuned configuration has both higher 'success_vote' (91% vs. 87%) and lower average inference cost ($0.015 vs. $0.037 per instance).

A developer could use AutoGen to tune the configuration to satisfy the target inference budget while maximizing the value out of it.
"""
logger.info("We find that the 'success_vote' metric is increased at the cost of exceeding the inference budget. But the tuned configuration has both higher 'success_vote' (91% vs. 87%) and lower average inference cost ($0.015 vs. $0.037 per instance).")

logger.info("\n\n[DONE]", bright=True)