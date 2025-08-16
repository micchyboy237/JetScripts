from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.retrieve_utils import TEXT_FORMATS
from jet.logger import CustomLogger
import autogen
import chromadb
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using RetrieveChat for Retrieve Augmented Code Generation and Question Answering

AutoGen offers conversable agents powered by LLM, tool or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

RetrieveChat is a conversational system for retrieval-augmented code generation and question answering. In this notebook, we demonstrate how to utilize RetrieveChat to generate code and answer questions based on customized documentations that are not present in the LLM's training dataset. RetrieveChat uses the `AssistantAgent` and `RetrieveUserProxyAgent`, which is similar to the usage of `AssistantAgent` and `UserProxyAgent` in other notebooks (e.g., [Automated Task Solving with Code Generation, Execution & Debugging](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_auto_feedback_from_code_execution.ipynb)). Essentially, `RetrieveUserProxyAgent` implement a different auto-reply mechanism corresponding to the RetrieveChat prompts.

## Table of Contents
We'll demonstrate six examples of using RetrieveChat for code generation and question answering:

- [Example 1: Generate code based off docstrings w/o human feedback](#example-1)
- [Example 2: Answer a question based off docstrings w/o human feedback](#example-2)
- [Example 3: Generate code based off docstrings w/ human feedback](#example-3)
- [Example 4: Answer a question based off docstrings w/ human feedback](#example-4)
- [Example 5: Solve comprehensive QA problems with RetrieveChat's unique feature `Update Context`](#example-5)
- [Example 6: Solve comprehensive QA problems with customized prompt and few-shot learning](#example-6)


````{=mdx}
:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:

```bash
pip install autogen[retrievechat] flaml[automl]
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````

## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("# Using RetrieveChat for Retrieve Augmented Code Generation and Question Answering")





config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

assert len(config_list) > 0
logger.debug("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````

## Construct agents for RetrieveChat

We start by initializing the `AssistantAgent` and `RetrieveUserProxyAgent`. The system message needs to be set to "You are a helpful assistant." for AssistantAgent. The detailed instructions are given in the user message. Later we will use the `RetrieveUserProxyAgent.message_generator` to combine the instructions and a retrieval augmented generation task for an initial prompt to be sent to the LLM assistant.
"""
logger.info("## Construct agents for RetrieveChat")

logger.debug("Accepted file formats for `docs_path`:")
logger.debug(TEXT_FORMATS)

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
        ],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "vector_db": "chroma",
        "overwrite": False,  # set to True if you want to overwrite an existing collection
        "get_or_create": True,  # set to False if don't want to reuse an existing collection
    },
    code_execution_config=False,  # set to False if you don't want to execute the code
)

"""
### Example 1

[Back to top](#table-of-contents)

Use RetrieveChat to help generate sample code and automatically run the code and fix errors if there is any.

Problem: Which API should I use if I want to use FLAML for a classification task and I want to train the model in 30 seconds. Use spark to parallel the training. Force cancel jobs if time limit is reached.
"""
logger.info("### Example 1")

assistant.reset()

code_problem = "How can I use FLAML to perform a classification task and use spark to do parallel training. Train 30 seconds and force cancel jobs if time limit is reached."
chat_result = ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem=code_problem, search_string="spark"
)  # search_string is used as an extra filter for the embeddings search, in this case, we only want to search documents that contain "spark".

"""
### Example 2

[Back to top](#table-of-contents)

Use RetrieveChat to answer a question that is not related to code generation.

Problem: Who is the author of FLAML?
"""
logger.info("### Example 2")

assistant.reset()

qa_problem = "Who is the author of FLAML?"
chat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=qa_problem)

"""
### Example 3

[Back to top](#table-of-contents)

Use RetrieveChat to help generate sample code and ask for human-in-loop feedbacks.

Problem: how to build a time series forecasting model for stock price using FLAML?
"""
logger.info("### Example 3")

assistant.reset()

ragproxyagent.human_input_mode = "ALWAYS"
code_problem = "how to build a time series forecasting model for stock price using FLAML?"
chat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=code_problem)

"""
### Example 4

[Back to top](#table-of-contents)

Use RetrieveChat to answer a question and ask for human-in-loop feedbacks.

Problem: Is there a function named `tune_automl` in FLAML?
"""
logger.info("### Example 4")

assistant.reset()

ragproxyagent.human_input_mode = "ALWAYS"
qa_problem = "Is there a function named `tune_automl` in FLAML?"
chat_result = ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem=qa_problem
)  # type "exit" to exit the conversation

"""
### Example 5

[Back to top](#table-of-contents)

Use RetrieveChat to answer questions for [NaturalQuestion](https://ai.google.com/research/NaturalQuestions) dataset.

First, we will create a new document collection which includes all the contextual corpus. Then, we will choose some questions and utilize RetrieveChat to answer them. For this particular example, we will be using the `gpt-3.5-turbo` model, and we will demonstrate RetrieveChat's feature of automatically updating context in case the documents retrieved do not contain sufficient information.
"""
logger.info("### Example 5")

config_list[0]["model"] = "gpt-35-turbo"  # change model to gpt-35-turbo

corpus_file = "https://huggingface.co/datasets/thinkall/NaturalQuestionsQA/resolve/main/corpus.txt"

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "qa",
        "docs_path": corpus_file,
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "natural-questions",
        "chunk_mode": "one_line",
        "embedding_model": "all-MiniLM-L6-v2",
    },
)

queries = """{"_id": "ce2342e1feb4e119cb273c05356b33309d38fa132a1cbeac2368a337e38419b8", "text": "what is non controlling interest on balance sheet", "metadata": {"answer": ["the portion of a subsidiary corporation 's stock that is not owned by the parent corporation"]}}
{"_id": "3a10ff0e520530c0aa33b2c7e8d989d78a8cd5d699201fc4b13d3845010994ee", "text": "how many episodes are in chicago fire season 4", "metadata": {"answer": ["23"]}}
{"_id": "fcdb6b11969d5d3b900806f52e3d435e615c333405a1ff8247183e8db6246040", "text": "what are bulls used for on a farm", "metadata": {"answer": ["breeding", "as work oxen", "slaughtered for meat"]}}
{"_id": "26c3b53ec44533bbdeeccffa32e094cfea0cc2a78c9f6a6c7a008ada1ad0792e", "text": "has been honoured with the wisden leading cricketer in the world award for 2016", "metadata": {"answer": ["Virat Kohli"]}}
{"_id": "0868d0964c719a52cbcfb116971b0152123dad908ac4e0a01bc138f16a907ab3", "text": "who carried the usa flag in opening ceremony", "metadata": {"answer": ["Erin Hamlin"]}}
"""
queries = [json.loads(line) for line in queries.split("\n") if line]
questions = [q["text"] for q in queries]
answers = [q["metadata"]["answer"] for q in queries]
logger.debug(questions)
logger.debug(answers)

for i in range(len(questions)):
    logger.debug(f"\n\n>>>>>>>>>>>>  Below are outputs of Case {i+1}  <<<<<<<<<<<<\n\n")

    assistant.reset()

    qa_problem = questions[i]
    chat_result = ragproxyagent.initiate_chat(
        assistant, message=ragproxyagent.message_generator, problem=qa_problem, n_results=30
    )

"""
In this example, questions were directly selected from the dataset. RetrieveChat was able to answer the questions correctly in the first attempt as the retrieved context contained the necessary information in the first two cases. However, in the last three cases, the context with the highest similarity to the question embedding did not contain the required information to answer the question. As a result, the LLM model responded with `UPDATE CONTEXT`. With the unique and innovative ability to update context in RetrieveChat, the agent automatically updated the context and sent it to the LLM model again. After several rounds of this process, the agent was able to generate the correct answer to the questions.

### Example 6

[Back to top](#table-of-contents)

Use RetrieveChat to answer multi-hop questions for [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) dataset with customized prompt and few-shot learning.

First, we will create a new document collection which includes all the contextual corpus. Then, we will choose some questions and utilize RetrieveChat to answer them. For this particular example, we will be using the `gpt-3.5-turbo` model, and we will demonstrate RetrieveChat's feature of automatically updating context in case the documents retrieved do not contain sufficient information. Moreover, we'll demonstrate how to use customized prompt and few-shot learning to address tasks that are not pre-defined in RetrieveChat.
"""
logger.info("### Example 6")

PROMPT_MULTIHOP = """You're a retrieve augmented chatbot. You answer user's questions based on your own knowledge and the context provided by the user. You must think step-by-step.
First, please learn the following examples of context and question pairs and their corresponding answers.

Context:
Kurram Garhi: Kurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000.
Trojkrsti: Trojkrsti is a village in Municipality of Prilep, Republic of Macedonia.
Q: Are both Kurram Garhi and Trojkrsti located in the same country?
A: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.


Context:
Early Side of Later: Early Side of Later is the third studio album by English singer- songwriter Matt Goss. It was released on 21 June 2004 by Concept Music and reached No. 78 on the UK Albums Chart.
What's Inside: What's Inside is the fourteenth studio album by British singer- songwriter Joan Armatrading.
Q: Which album was released earlier, What'S Inside or Cassandra'S Dream (Album)?
A: What's Inside was released in the year 1995. Cassandra's Dream (album) was released in the year 2008. Thus, of the two, the album to release earlier is What's Inside. So the answer is: What's Inside.


Context:
Maria Alexandrovna (Marie of Hesse): Maria Alexandrovna , born Princess Marie of Hesse and by Rhine (8 August 1824 – 3 June 1880) was Empress of Russia as the first wife of Emperor Alexander II.
Grand Duke Alexei Alexandrovich of Russia: Grand Duke Alexei Alexandrovich of Russia,(Russian: Алексей Александрович; 14 January 1850 (2 January O.S.) in St. Petersburg – 14 November 1908 in Paris) was the fifth child and the fourth son of Alexander II of Russia and his first wife Maria Alexandrovna (Marie of Hesse).
Q: What is the cause of death of Grand Duke Alexei Alexandrovich Of Russia's mother?
A: The mother of Grand Duke Alexei Alexandrovich of Russia is Maria Alexandrovna. Maria Alexandrovna died from tuberculosis. So the answer is: tuberculosis.


Context:
Laughter in Hell: Laughter in Hell is a 1933 American Pre-Code drama film directed by Edward L. Cahn and starring Pat O'Brien. The film's title was typical of the sensationalistic titles of many Pre-Code films.
Edward L. Cahn: Edward L. Cahn (February 12, 1899 – August 25, 1963) was an American film director.
Q: When did the director of film Laughter In Hell die?
A: The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963. So the answer is: August 25, 1963.

Second, please complete the answer by thinking step-by-step.

Context:
{input_context}
Q: {input_question}
A:
"""

corpus_file = "https://huggingface.co/datasets/thinkall/2WikiMultihopQA/resolve/main/corpus.txt"

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",
        "docs_path": corpus_file,
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "2wikimultihopqa",
        "chunk_mode": "one_line",
        "embedding_model": "all-MiniLM-L6-v2",
        "customized_prompt": PROMPT_MULTIHOP,
        "customized_answer_prefix": "the answer is",
    },
)

queries = """{"_id": "61a46987092f11ebbdaeac1f6bf848b6", "text": "Which film came out first, Blind Shaft or The Mask Of Fu Manchu?", "metadata": {"answer": ["The Mask Of Fu Manchu"]}}
{"_id": "a7b9672009c311ebbdb0ac1f6bf848b6", "text": "Are North Marion High School (Oregon) and Seoul High School both located in the same country?", "metadata": {"answer": ["no"]}}
"""
queries = [json.loads(line) for line in queries.split("\n") if line]
questions = [q["text"] for q in queries]
answers = [q["metadata"]["answer"] for q in queries]
logger.debug(questions)
logger.debug(answers)

for i in range(len(questions)):
    logger.debug(f"\n\n>>>>>>>>>>>>  Below are outputs of Case {i+1}  <<<<<<<<<<<<\n\n")

    assistant.reset()

    qa_problem = questions[i]
    chat_result = ragproxyagent.initiate_chat(
        assistant, message=ragproxyagent.message_generator, problem=qa_problem, n_results=10
    )

logger.info("\n\n[DONE]", bright=True)