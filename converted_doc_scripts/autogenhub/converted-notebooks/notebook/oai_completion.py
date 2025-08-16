from flaml import tune
from functools import partial
from jet.logger import CustomLogger
import autogen
import datasets
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


"""
<a href="https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/oai_completion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Contributions to this project, i.e., https://github.com/autogenhub/autogen, are licensed under the Apache License, Version 2.0 (Apache-2.0).
Copyright (c) 2023 - 2024, Owners of https://github.com/autogenhub
SPDX-License-Identifier: Apache-2.0
Portions derived from  https://github.com/microsoft/autogen under the MIT License.
SPDX-License-Identifier: MIT
Copyright (c) Microsoft Corporation. All rights reserved. 

Licensed under the MIT License.

# Use AutoGen to Tune Ollama Models

AutoGen offers a cost-effective hyperparameter optimization technique [EcoOptiGen](https://arxiv.org/abs/2303.04673) for tuning Large Language Models. The research study finds that tuning hyperparameters can significantly improve the utility of LLMs.
Please find documentation about this feature [here](/docs/Use-Cases/AutoGen#enhanced-inference).

In this notebook, we tune Ollama models for code generation. We use [the HumanEval benchmark](https://huggingface.co/datasets/openai_humaneval) released by Ollama for synthesizing programs from docstrings.

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install with the [blendsearch] option:
```bash
pip install autogen[blendsearch]
```
"""
logger.info("# Use AutoGen to Tune Ollama Models")



"""
## Set your API Endpoint

* The [`config_list_openai_aoai`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_openai_aoai) function tries to create a list of configurations using Azure Ollama endpoints and Ollama endpoints. It assumes the api keys and api bases are stored in the corresponding environment variables or local txt files:
#   - Ollama API key: os.environ["OPENAI_API_KEY"] or `openai_api_key_file="key_openai.txt"`.
#   - Azure Ollama API key: os.environ["AZURE_OPENAI_API_KEY"] or `aoai_api_key_file="key_aoai.txt"`. Multiple keys can be stored, one per line.
  - Azure Ollama API base: os.environ["AZURE_OPENAI_API_BASE"] or `aoai_api_base_file="base_aoai.txt"`. Multiple bases can be stored, one per line.
* The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file. It first looks for the environment variable `env_or_file`, which must be a valid json string. If that variable is not found, it looks for a json file with the same name. It filters the configs by filter_dict.

It's OK to have only the Ollama API key, or only the Azure Ollama API key + base. If you open this notebook in colab, you can upload your files by clicking the file icon on the left panel and then choosing "upload file" icon.
"""
logger.info("## Set your API Endpoint")




endpoint_list = autogen.config_list_openai_aoai()

config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={
        "model": {
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "chatgpt-35-turbo-0301",
            "gpt-35-turbo-v0301",
            "gpt",
        },
    },
)

"""
If you don't use the two provided utility functions above, you can define the lists in other ways you prefer.

## Load dataset

First, we load the humaneval dataset. The dataset contains 164 examples. We use the first 20 for tuning the generation hyperparameters and the remaining for evaluation. In each example, the "prompt" is the prompt string for eliciting the code generation (renamed into "definition"), "test" is the Python code for unit test for the example, and "entry_point" is the function name to be tested.
"""
logger.info("## Load dataset")

seed = 41
data = datasets.load_dataset("openai_humaneval")["test"].shuffle(seed=seed)
n_tune_data = 20
tune_data = [
    {
        "definition": data[x]["prompt"],
        "test": data[x]["test"],
        "entry_point": data[x]["entry_point"],
    }
    for x in range(n_tune_data)
]
test_data = [
    {
        "definition": data[x]["prompt"],
        "test": data[x]["test"],
        "entry_point": data[x]["entry_point"],
    }
    for x in range(n_tune_data, len(data))
]

"""
Check a tuning example:
"""
logger.info("Check a tuning example:")

logger.debug(tune_data[1]["definition"])

"""
Here is one example of the unit test code for verifying the correctness of the generated code:
"""
logger.info("Here is one example of the unit test code for verifying the correctness of the generated code:")

logger.debug(tune_data[1]["test"])

"""
## Define Success Metric

Before we start tuning, we need to define the success metric we want to optimize. For each code generation task, we can use the model to generate multiple candidates, and then select one from them. If the final selected response can pass a unit test, we consider the task as successfully solved. Then we can define the mean success rate of a collection of tasks.
"""
logger.info("## Define Success Metric")

eval_with_generated_assertions = partial(
    autogen.code_utils.eval_function_completions,
    assertions=partial(autogen.code_utils.generate_assertions, config_list=config_list),
    use_docker=False,
)

"""
This function will first generate assertion statements for each problem. Then, it uses the assertions to select the generated responses.

## Use the tuning data to find a good configuration

AutoGen has provided an API for hyperparameter optimization of Ollama models: `autogen.Completion.tune` and to make a request with the tuned config: `autogen.Completion.create`.

For (local) reproducibility and cost efficiency, we cache responses from Ollama with a controllable seed.
"""
logger.info("## Use the tuning data to find a good configuration")

autogen.Completion.set_cache(seed)

"""
This will create a disk cache in ".cache/{seed}". You can change `cache_path_root` from ".cache" to a different path in `set_cache()`. The cache for different seeds are stored separately.

### Perform tuning

The tuning will take a while to finish, depending on the optimization budget. The tuning will be performed under the specified optimization budgets.

* `inference_budget` is the target average inference budget per instance in the benchmark. For example, 0.02 means the target inference budget is 0.02 dollars, which translates to 1000 tokens (input + output combined) if the text Davinci model is used.
* `optimization_budget` is the total budget allowed to perform the tuning. For example, 5 means 5 dollars are allowed in total, which translates to 250K tokens for the text Davinci model.
* `num_sumples` is the number of different hyperparameter configurations allowed to be tried. The tuning will stop after either num_samples trials or after optimization_budget dollars spent, whichever happens first. -1 means no hard restriction in the number of trials and the actual number is decided by `optimization_budget`.

Users can specify tuning data, optimization metric, optimization mode, evaluation function, search spaces, etc. The default search space is:

```python

default_search_space = {
    "model": tune.choice([
        "text-ada-001",
        "text-babbage-001",
        "text-davinci-003",
        "gpt-3.5-turbo",
        "gpt-4",
    ]),
    "temperature_or_top_p": tune.choice(
        [
            {"temperature": tune.uniform(0, 1)},
            {"top_p": tune.uniform(0, 1)},
        ]
    ),
    "max_tokens": tune.lograndint(50, 1000),
    "n": tune.randint(1, 100),
    "prompt": "{prompt}",
}
```

Users' input can override the default search space.
For example, the following code specifies three choices for the prompt and two choices of stop sequences. The default search space will be used for hyperparameters that don't appear in users' input. If you don't have access to gpt-4 or would like to modify the choice of models, you can provide a different search space for the model.
"""
logger.info("### Perform tuning")

config, analysis = autogen.Completion.tune(
    data=tune_data,  # the data for tuning
    metric="success",  # the metric to optimize
    mode="max",  # the optimization mode
    eval_func=eval_with_generated_assertions,  # the evaluation function to return the success metrics
    inference_budget=0.05,  # the inference budget (dollar per instance)
    optimization_budget=1,  # the optimization budget (dollar in total)
    num_samples=-1,
    prompt=[
        "{definition}",
        "# Python 3{definition}",
        "Complete the following Python function:{definition}",
    ],  # the prompt templates to choose from
    stop=[["\nclass", "\ndef", "\nif", "\nprint"], None],  # the stop sequences
    config_list=endpoint_list,  # optional: a list of endpoints to use
    allow_format_str_template=True,  # whether to allow format string template
)

"""
### Output tuning results

After the tuning, we can print out the config and the result found by autogen:
"""
logger.info("### Output tuning results")

logger.debug("optimized config", config)
logger.debug("best result on tuning data", analysis.best_result)

"""
### Request with the tuned config

We can apply the tuned config on the request for an example task:
"""
logger.info("### Request with the tuned config")

response = autogen.Completion.create(context=tune_data[1], config_list=endpoint_list, **config)
logger.debug(response)
logger.debug(eval_with_generated_assertions(autogen.Completion.extract_text(response), **tune_data[1]))

"""
### Evaluate the success rate on the test data

You can use `autogen.Completion.test` to evaluate the performance of an entire dataset with the tuned config. The following code will take a while to evaluate all the 144 test data instances. The cost is about $6 if you uncomment it and run it.
"""
logger.info("### Evaluate the success rate on the test data")



"""
The result will vary with the inference budget and optimization budget.
"""
logger.info("The result will vary with the inference budget and optimization budget.")

logger.info("\n\n[DONE]", bright=True)