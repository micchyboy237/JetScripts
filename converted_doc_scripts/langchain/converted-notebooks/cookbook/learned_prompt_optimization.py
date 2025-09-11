from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
import langchain_experimental.rl_chain as rl_chain
import logging
import os
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


"""
# Learned Prompt Variable Injection via RL

LLM prompts can be enhanced by injecting specific terms into template sentences. Selecting the right terms is crucial for obtaining high-quality responses. This notebook introduces automated prompt engineering through term injection using Reinforcement Learning with VowpalWabbit.

The rl_chain (reinforcement learning chain) provides a way to automatically determine the best terms to inject without the need for fine-tuning the underlying foundational model.

For illustration, consider the scenario of a meal delivery service. We use LangChain to ask customers, like Tom, about their dietary preferences and recommend suitable meals from our extensive menu. The rl_chain selects a meal based on user preferences, injects it into a prompt template, and forwards the prompt to an LLM. The LLM's response, which is a personalized recommendation, is then returned to the user.

The example laid out below is a toy example to demonstrate the applicability of the concept. Advanced options and explanations are provided at the end.
"""
logger.info("# Learned Prompt Variable Injection via RL")


meals = [
    "Beef Enchiladas with Feta cheese. Mexican-Greek fusion",
    "Chicken Flatbreads with red sauce. Italian-Mexican fusion",
    "Veggie sweet potato quesadillas with vegan cheese",
    "One-Pan Tortelonni bake with peppers and onions",
]


llm = ChatOllama(model="llama3.2")

"""
##### Initialize the RL chain with provided defaults

The prompt template which will be used to query the LLM needs to be defined.
It can be anything, but here `{meal}` is being used and is going to be replaced by one of the meals above, the RL chain will try to pick and inject the best meal
"""
logger.info("##### Initialize the RL chain with provided defaults")


PROMPT_TEMPLATE = """Here is the description of a meal: "{meal}".

Embed the meal into the given text: "{text_to_personalize}".

Prepend a personalized message including the user's name "{user}"
    and their preference "{preference}".

Make it sound good.
"""

PROMPT = PromptTemplate(
    input_variables=["meal", "text_to_personalize", "user", "preference"],
    template=PROMPT_TEMPLATE,
)

"""
Next the RL chain's PickBest chain is being initialized. We must provide the llm of choice and the defined prompt. As the name indicates, the chain's goal is to Pick the Best of the meals that will be provided, based on some criteria.
"""
logger.info("Next the RL chain's PickBest chain is being initialized. We must provide the llm of choice and the defined prompt. As the name indicates, the chain's goal is to Pick the Best of the meals that will be provided, based on some criteria.")


chain = rl_chain.PickBest.from_llm(llm=llm, prompt=PROMPT)

"""
Once the chain is setup I am going to call it with the meals I want to be selected from, and some context based on which the chain will select a meal.
"""
logger.info("Once the chain is setup I am going to call it with the meals I want to be selected from, and some context based on which the chain will select a meal.")

response = chain.run(
    meal=rl_chain.ToSelectFrom(meals),
    user=rl_chain.BasedOn("Tom"),
    preference=rl_chain.BasedOn(["Vegetarian", "regular dairy is ok"]),
    text_to_personalize="This is the weeks specialty dish, our master chefs \
        believe you will love it!",
)

logger.debug(response["response"])

"""
## What is the chain doing

Here's a step-by-step breakdown of the RL chain's operations:

1. Accept the list of meals.
2. Consider the user and their dietary preferences.
3. Based on this context, select an appropriate meal.
4. Automatically evaluate the appropriateness of the meal choice.
5. Inject the selected meal into the prompt and submit it to the LLM.
6. Return the LLM's response to the user.

Technically, the chain achieves this by employing a contextual bandit reinforcement learning model, specifically utilizing the [VowpalWabbit](https://github.com/VowpalWabbit/vowpal_wabbit) ML library.

Initially, since the RL model is untrained, it might opt for random selections that don't necessarily align with a user's preferences. However, as it gains more exposure to the user's choices and feedback, it should start to make better selections (or quickly learn a good one and just pick that!).
"""
logger.info("## What is the chain doing")

for _ in range(5):
    try:
        response = chain.run(
            meal=rl_chain.ToSelectFrom(meals),
            user=rl_chain.BasedOn("Tom"),
            preference=rl_chain.BasedOn(["Vegetarian", "regular dairy is ok"]),
            text_to_personalize="This is the weeks specialty dish, our master chefs believe you will love it!",
        )
    except Exception as e:
        logger.debug(e)
    logger.debug(response["response"])
    logger.debug()

"""
## How is the chain learning

It's important to note that while the RL model can make sophisticated selections, it doesn't inherently recognize concepts like "vegetarian" or understand that "beef enchiladas" aren't vegetarian-friendly. Instead, it leverages the LLM to ground its choices in common sense.

The way the chain is learning that Tom prefers vegetarian meals is via an AutoSelectionScorer that is built into the chain. The scorer will call the LLM again and ask it to evaluate the selection (`ToSelectFrom`) using the information wrapped in (`BasedOn`).

You can set `set_debug(True)` if you want to see the details of the auto-scorer, but you can also define the scoring prompt yourself.
"""
logger.info("## How is the chain learning")

scoring_criteria_template = (
    "Given {preference} rank how good or bad this selection is {meal}"
)

chain = rl_chain.PickBest.from_llm(
    llm=llm,
    prompt=PROMPT,
    selection_scorer=rl_chain.AutoSelectionScorer(
        llm=llm, scoring_criteria_template_str=scoring_criteria_template
    ),
)

"""
If you want to examine the score and other selection metadata you can by examining the metadata object returned by the chain
"""
logger.info("If you want to examine the score and other selection metadata you can by examining the metadata object returned by the chain")

response = chain.run(
    meal=rl_chain.ToSelectFrom(meals),
    user=rl_chain.BasedOn("Tom"),
    preference=rl_chain.BasedOn(["Vegetarian", "regular dairy is ok"]),
    text_to_personalize="This is the weeks specialty dish, our master chefs believe you will love it!",
)
logger.debug(response["response"])
selection_metadata = response["selection_metadata"]
logger.debug(
    f"selected index: {selection_metadata.selected.index}, score: {selection_metadata.selected.score}"
)

"""
In a more realistic scenario it is likely that you have a well defined scoring function for what was selected. For example, you might be doing few-shot prompting and want to select prompt examples for a natural language to sql translation task. In that case the scorer could be: did the sql that was generated run in an sql engine? In that case you want to plugin a scoring function. In the example below I will just check if the meal picked was vegetarian or not.
"""
logger.info("In a more realistic scenario it is likely that you have a well defined scoring function for what was selected. For example, you might be doing few-shot prompting and want to select prompt examples for a natural language to sql translation task. In that case the scorer could be: did the sql that was generated run in an sql engine? In that case you want to plugin a scoring function. In the example below I will just check if the meal picked was vegetarian or not.")


class CustomSelectionScorer(rl_chain.SelectionScorer):
    def score_response(
        self, inputs, llm_response: str, event: rl_chain.PickBestEvent
    ) -> float:
        logger.debug(event.based_on)
        logger.debug(event.to_select_from)

        selected_meal = event.to_select_from["meal"][event.selected.index]
        logger.debug(f"selected meal: {selected_meal}")

        if "Tom" in event.based_on["user"]:
            if "Vegetarian" in event.based_on["preference"]:
                if "Chicken" in selected_meal or "Beef" in selected_meal:
                    return 0.0
                else:
                    return 1.0
            else:
                if "Chicken" in selected_meal or "Beef" in selected_meal:
                    return 1.0
                else:
                    return 0.0
        else:
            raise NotImplementedError("I don't know how to score this user")


chain = rl_chain.PickBest.from_llm(
    llm=llm,
    prompt=PROMPT,
    selection_scorer=CustomSelectionScorer(),
)

response = chain.run(
    meal=rl_chain.ToSelectFrom(meals),
    user=rl_chain.BasedOn("Tom"),
    preference=rl_chain.BasedOn(["Vegetarian", "regular dairy is ok"]),
    text_to_personalize="This is the weeks specialty dish, our master chefs believe you will love it!",
)

"""
## How can I track the chains progress

You can track the chains progress by using the metrics mechanism provided. I am going to expand the users to Tom and Anna, and extend the scoring function. I am going to initialize two chains, one with the default learning policy and one with a built-in random policy (i.e. selects a meal randomly), and plot their scoring progress.
"""
logger.info("## How can I track the chains progress")


class CustomSelectionScorer(rl_chain.SelectionScorer):
    def score_preference(self, preference, selected_meal):
        if "Vegetarian" in preference:
            if "Chicken" in selected_meal or "Beef" in selected_meal:
                return 0.0
            else:
                return 1.0
        else:
            if "Chicken" in selected_meal or "Beef" in selected_meal:
                return 1.0
            else:
                return 0.0

    def score_response(
        self, inputs, llm_response: str, event: rl_chain.PickBestEvent
    ) -> float:
        selected_meal = event.to_select_from["meal"][event.selected.index]

        if "Tom" in event.based_on["user"]:
            return self.score_preference(event.based_on["preference"], selected_meal)
        elif "Anna" in event.based_on["user"]:
            return self.score_preference(event.based_on["preference"], selected_meal)
        else:
            raise NotImplementedError("I don't know how to score this user")


chain = rl_chain.PickBest.from_llm(
    llm=llm,
    prompt=PROMPT,
    selection_scorer=CustomSelectionScorer(),
    metrics_step=5,
    metrics_window_size=5,  # rolling window average
)

random_chain = rl_chain.PickBest.from_llm(
    llm=llm,
    prompt=PROMPT,
    selection_scorer=CustomSelectionScorer(),
    metrics_step=5,
    metrics_window_size=5,  # rolling window average
    # set the random policy instead of default
    policy=rl_chain.PickBestRandomPolicy,
)

for _ in range(20):
    try:
        chain.run(
            meal=rl_chain.ToSelectFrom(meals),
            user=rl_chain.BasedOn("Tom"),
            preference=rl_chain.BasedOn(["Vegetarian", "regular dairy is ok"]),
            text_to_personalize="This is the weeks specialty dish, our master chefs believe you will love it!",
        )
        random_chain.run(
            meal=rl_chain.ToSelectFrom(meals),
            user=rl_chain.BasedOn("Tom"),
            preference=rl_chain.BasedOn(["Vegetarian", "regular dairy is ok"]),
            text_to_personalize="This is the weeks specialty dish, our master chefs believe you will love it!",
        )

        chain.run(
            meal=rl_chain.ToSelectFrom(meals),
            user=rl_chain.BasedOn("Anna"),
            preference=rl_chain.BasedOn(["Loves meat", "especially beef"]),
            text_to_personalize="This is the weeks specialty dish, our master chefs believe you will love it!",
        )
        random_chain.run(
            meal=rl_chain.ToSelectFrom(meals),
            user=rl_chain.BasedOn("Anna"),
            preference=rl_chain.BasedOn(["Loves meat", "especially beef"]),
            text_to_personalize="This is the weeks specialty dish, our master chefs believe you will love it!",
        )
    except Exception as e:
        logger.debug(e)

"""
The RL chain converges to the fact that Anna prefers beef and Tom is vegetarian. The random chain picks at random, and so will send beef to vegetarians half the time.
"""
logger.info("The RL chain converges to the fact that Anna prefers beef and Tom is vegetarian. The random chain picks at random, and so will send beef to vegetarians half the time.")


chain.metrics.to_pandas()["score"].plot(label="default learning policy")
random_chain.metrics.to_pandas()["score"].plot(label="random selection policy")
plt.legend()

logger.debug(
    f"The final average score for the default policy, calculated over a rolling window, is: {chain.metrics.to_pandas()['score'].iloc[-1]}"
)
logger.debug(
    f"The final average score for the random policy, calculated over a rolling window, is: {random_chain.metrics.to_pandas()['score'].iloc[-1]}"
)

"""
There is a bit of randomness involved in the rl_chain's selection since the chain explores the selection space in order to learn the world as best as it can (see details of default exploration algorithm used [here](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-Exploration-with-SquareCB)), but overall, default chain policy should be doing better than random as it learns

## Advanced options

The RL chain is highly configurable in order to be able to adjust to various selection scenarios. If you want to learn more about the ML library that powers it please take a look at tutorials [here](https://vowpalwabbit.org/)

| Section | Description | Example / Usage |
|---------|-------------|-----------------|
| [**Change Chain Logging Level**](#change-chain-logging-level) | Change the logging level for the RL chain. | `logger.setLevel(logging.INFO)` |
| [**Featurization**](#featurization) | Adjusts the input to the RL chain. Can set auto-embeddings ON for more complex embeddings. | `chain = rl_chain.PickBest.from_llm(auto_embed=True, [...])` |
| [**Learned Policy to Learn Asynchronously**](#learned-policy-to-learn-asynchronously) | Score asynchronously if user input is needed for scoring. | `chain.update_with_delayed_score(score=<the score>, chain_response=response)` |
| [**Store Progress of Learned Policy**](#store-progress-of-learned-policy) | Option to store the progress of the variable injection learned policy. | `chain.save_progress()` |
| [**Stop Learning of Learned Policy**](#stop-learning-of-learned-policy) | Toggle the RL chain's learned policy updates ON/OFF. | `chain.deactivate_selection_scorer()` |
| [**Set a Different Policy**](#set-a-different-policy) | Choose between different policies: default, random, or custom. | Custom policy creation at chain creation time. |
| [**Different Exploration Algorithms and Options for Default Learned Policy**](#different-exploration-algorithms-and-options-for-the-default-learned-policy) | Set different exploration algorithms and hyperparameters for `VwPolicy`. | `vw_cmd = ["--cb_explore_adf", "--quiet", "--squarecb", "--interactions=::"]` |
| [**Learn Policy's Data Logs**](#learned-policys-data-logs) | Store and examine `VwPolicy`'s data logs. | `chain = rl_chain.PickBest.from_llm(vw_logs=<path to log FILE>, [...])` |
| [**Other Advanced Featurization Options**](#other-advanced-featurization-options) | Specify advanced featurization options for the RL chain. | `age = rl_chain.BasedOn("age:32")` |
| [**More Info on Auto or Custom SelectionScorer**](#more-info-on-auto-or-custom-selectionscorer) | Dive deeper into how selection scoring is determined. | `selection_scorer=rl_chain.AutoSelectionScorer(llm=llm, scoring_criteria_template_str=scoring_criteria_template)` |

### change chain logging level

```
logger = logging.getLogger("rl_chain")
logger.setLevel(logging.INFO)
```

### featurization

#### auto_embed

By default the input to the rl chain (`ToSelectFrom`, `BasedOn`) is not tampered with. This might  not be sufficient featurization, so based on how complex the scenario is you can set auto-embeddings to ON

`chain = rl_chain.PickBest.from_llm(auto_embed=True, [...])`

This will produce more complex embeddings and featurizations of the inputs, likely accelerating RL chain learning, albeit at the cost of increased runtime.

By default, [sbert.net's sentence_transformers's ](https://www.sbert.net/docs/pretrained_models.html#model-overview) `all-mpnet-base-v2` model will be used for these embeddings but you can set a different embeddings model by initializing the chain with it as shown in this example. You could also set an entirely different embeddings encoding object, as long as it has an `encode()` function that returns a list of the encodings.

```

chain = rl_chain.PickBest.from_llm(
    [...]
    feature_embedder=rl_chain.PickBestFeatureEmbedder(
        auto_embed=True,
        model=SentenceTransformer("all-mpnet-base-v2")
    )
)
```

#### explicitly defined embeddings

Another option is to define what inputs you think should be embedded manually:
- `auto_embed = False`
- Can wrap individual variables in `rl_chain.Embed()` or `rl_chain.EmbedAndKeep()` e.g. `user = rl_chain.BasedOn(rl_chain.Embed("Tom"))`

#### custom featurization

Another final option is to define and set a custom featurization/embedder class that returns a valid input for the learned policy.

## learned policy to learn asynchronously

If to score the result you need input from the user (e.g. my application showed Tom the selected meal and Tom clicked on it, but Anna did not), then the scoring can be done asynchronously. The way to do that is:

- set `selection_scorer=None` on the chain creation OR call `chain.deactivate_selection_scorer()`
- call the chain for a specific input
- keep the chain's response (`response = chain.run([...])`)
- once you have determined the score of the response/chain selection call the chain with it: `chain.update_with_delayed_score(score=<the score>, chain_response=response)`

### store progress of learned policy

Since the variable injection learned policy evolves over time, there is the option to store its progress and continue learning. This can be done by calling:

`chain.save_progress()`

which will store the rl chain's learned policy in a file called `latest.vw`. It will also store it in a file with a timestamp. That way, if `save_progress()` is called more than once, multiple checkpoints will be created, but the latest one will always be in `latest.vw`

Next time the chain is loaded, the chain will look for a file called `latest.vw` and if the file exists it will be loaded into the chain and the learning will continue from there.

By default the rl chain model checkpoints will be stored in the current directory but you can specify the save/load location at chain creation time:

`chain = rl_chain.PickBest.from_llm(model_save_dir=<path to dir>, [...])`

### stop learning of learned policy

If you want the rl chain's learned policy to stop updating you can turn it off/on:

`chain.deactivate_selection_scorer()` and `chain.activate_selection_scorer()`

### set a different policy

There are two policies currently available:

- default policy: `VwPolicy` which learns a [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) [Contextual Bandit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms) model

- random policy: `RandomPolicy` which doesn't learn anything and just selects a value randomly. this policy can be used to compare other policies with a random baseline one.

- custom policies: a custom policy could be created and set at chain creation time

### different exploration algorithms and options for the default learned policy

The default `VwPolicy` is initialized with some default arguments. The default exploration algorithm is [SquareCB](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-Exploration-with-SquareCB) but other Contextual Bandit exploration algorithms can be set, and other hyper parameters can be tuned (see [here](https://vowpalwabbit.org/docs/vowpal_wabbit/python/9.6.0/command_line_args.html) for available options).

`vw_cmd = ["--cb_explore_adf", "--quiet", "--squarecb", "--interactions=::"]`

`chain = rl_chain.PickBest.from_llm(vw_cmd = vw_cmd, [...])`

### learned policy's data logs

The `VwPolicy`'s data files can be stored and examined or used to do [off policy evaluation](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/off_policy_evaluation.html) for hyper parameter tuning.

The way to do this is to set a log file path to `vw_logs` on chain creation:

`chain = rl_chain.PickBest.from_llm(vw_logs=<path to log FILE>, [...])`

### other advanced featurization options

Explicitly numerical features can be provided with a colon separator:
`age = rl_chain.BasedOn("age:32")`

`ToSelectFrom` can be a bit more complex if the scenario demands it, instead of being a list of strings it can be:
- a list of list of strings:
    ```
    meal = rl_chain.ToSelectFrom([
        ["meal 1 name", "meal 1 description"],
        ["meal 2 name", "meal 2 description"]
    ])
    ```
- a list of dictionaries:
    ```
    meal = rl_chain.ToSelectFrom([
        {"name":"meal 1 name", "description" : "meal 1 description"},
        {"name":"meal 2 name", "description" : "meal 2 description"}
    ])
    ```
- a list of dictionaries containing lists:
    ```
    meal = rl_chain.ToSelectFrom([
        {"name":["meal 1", "complex name"], "description" : "meal 1 description"},
        {"name":["meal 2", "complex name"], "description" : "meal 2 description"}
    ])
    ```

`BasedOn` can also take a list of strings:
```
user = rl_chain.BasedOn(["Tom Joe", "age:32", "state of california"])
```

there is no dictionary provided since multiple variables can be supplied wrapped in `BasedOn`

Storing the data logs into a file allows the examination of what different inputs do to the data format.

### More info on Auto or Custom SelectionScorer

It is very important to get the selection scorer right since the policy uses it to learn. It determines what is called the reward in reinforcement learning, and more specifically in our Contextual Bandits setting.

The general advice is to keep the score between [0, 1], 0 being the worst selection, 1 being the best selection from the available `ToSelectFrom` variables, based on the `BasedOn` variables, but should be adjusted if the need arises.

In the examples provided above, the AutoSelectionScorer is set mostly to get users started but in real world scenarios it will most likely not be an adequate scorer function.

The example also provided the option to change part of the scoring prompt template that the AutoSelectionScorer used to determine whether a selection was good or not:

```
scoring_criteria_template = "Given {preference} rank how good or bad this selection is {meal}"
chain = rl_chain.PickBest.from_llm(
    llm=llm,
    prompt=PROMPT,
    selection_scorer=rl_chain.AutoSelectionScorer(llm=llm, scoring_criteria_template_str=scoring_criteria_template),
)

```

Internally the AutoSelectionScorer adjusted the scoring prompt to make sure that the llm scoring returned a single float.

However, if needed, a FULL scoring prompt can also be provided:
"""
logger.info("## Advanced options")


set_debug(True)

REWARD_PROMPT_TEMPLATE = """

Given {preference} rank how good or bad this selection is {meal}

IMPORTANT: you MUST return a single number between -1 and 1, -1 being bad, 1 being good

"""


REWARD_PROMPT = PromptTemplate(
    input_variables=["preference", "meal"],
    template=REWARD_PROMPT_TEMPLATE,
)

chain = rl_chain.PickBest.from_llm(
    llm=llm,
    prompt=PROMPT,
    selection_scorer=rl_chain.AutoSelectionScorer(
        llm=llm, prompt=REWARD_PROMPT),
)

chain.run(
    meal=rl_chain.ToSelectFrom(meals),
    user=rl_chain.BasedOn("Tom"),
    preference=rl_chain.BasedOn(["Vegetarian", "regular dairy is ok"]),
    text_to_personalize="This is the weeks specialty dish, our master chefs believe you will love it!",
)

logger.info("\n\n[DONE]", bright=True)
