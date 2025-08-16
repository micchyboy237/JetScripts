from autogen import AssistantAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from flaml import AutoML
from jet.logger import CustomLogger
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from synapse.ml.fabric.credentials import get_openai_httpx_sync_client
import autogen
import flaml.visualization as fviz
import httpx
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tempfile
import time
import types


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## Use AutoGen in Microsoft Fabric

[AutoGen](https://github.com/autogenhub/autogen) offers conversable LLM agents, which can be used to solve various tasks with human or automatic feedback, including tasks that require using tools via code.
Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

[Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/get-started/microsoft-fabric-overview) is an all-in-one analytics solution for enterprises that covers everything from data movement to data science, Real-Time Analytics, and business intelligence. It offers a comprehensive suite of services, including data lake, data engineering, and data integration, all in one place. Its pre-built AI models include GPT-x models such as `gpt-4o`, `gpt-4-turbo`, `gpt-4`, `gpt-4-8k`, `gpt-4-32k`, `gpt-35-turbo`, `gpt-35-turbo-16k` and `gpt-35-turbo-instruct`, etc. It's important to note that the Azure Open AI service is not supported on trial SKUs and only paid SKUs (F64 or higher, or P1 or higher) are supported.

In this notebook, we demonstrate several examples:
- 0. How to access pre-built LLM endpoints with AutoGen in Microsoft Fabric.
- 1. How to use `AssistantAgent` and `UserProxyAgent` to write code and execute the code.
- 2. How to use `AssistantAgent` and `RetrieveUserProxyAgent` to do Retrieval Augmented Generation (RAG) for QA and Code Generation.
- 3. How to use `MultimodalConversableAgent` to chat with images.

#### Requirements

AutoGen requires `Python>=3.8`.

Also, this notebook depends on Microsoft Fabric pre-built LLM endpoints and Fabric runtime 1.2+. Running it elsewhere may encounter errors.

### Example 0

#### Work with openai<1

AutoGen can work with openai<1 in Microsoft Fabric. To access pre-built LLM endpoints with AutoGen, you can follow below example.

This example can run in Fabric runtime 1.2+.
"""
logger.info("## Use AutoGen in Microsoft Fabric")

# %pip install "pyautogen==0.1.14" "openai==0.28.1" -q

config_list = [
    {
        "model": "gpt-4o",
    },
]

llm_config = {
    "config_list": config_list,
    "temperature": 0,
    "timeout": 600,
}


agent = autogen.agentchat.ConversableAgent(
    name=config_list[0]["model"], llm_config=llm_config, max_consecutive_auto_reply=1, human_input_mode="NEVER"
)
userproxy = autogen.agentchat.ConversableAgent(
    name="user",
    max_consecutive_auto_reply=0,
    llm_config=False,
    default_auto_reply="TERMINATE",
    human_input_mode="NEVER",
)
userproxy.initiate_chat(recipient=agent, message="Tell me a quick joke.")

"""
#### Work with openai>=1

AutoGen can work with openai>=1 in Microsoft Fabric. To access pre-built LLM endpoints with AutoGen, you can follow below example.

This example and below examples can only run in Fabric runtime 1.3+.
"""
logger.info("#### Work with openai>=1")

# %pip install "autogen>0.2" "openai>1" -q




http_client = get_openai_httpx_sync_client()  # http_client is needed for openai>1
http_client.__deepcopy__ = types.MethodType(
    lambda self, memo: self, http_client
)  # https://autogenhub.github.io/autogen/docs/topics/llm_configuration#adding-http-client-in-llm_config-for-proxy\n",

config_list = [
    {
        "model": "gpt-4o",
        "http_client": http_client,
        "api_version": "2024-02-01",
        "api_type": "azure",
    },
]

llm_config = {
    "config_list": config_list,
    "temperature": 0,
}


agent = autogen.agentchat.ConversableAgent(
    name=config_list[0]["model"], llm_config=llm_config, max_consecutive_auto_reply=1, human_input_mode="NEVER"
)
userproxy = autogen.agentchat.ConversableAgent(
    name="user",
    max_consecutive_auto_reply=0,
    llm_config=False,
    default_auto_reply="TERMINATE",
    human_input_mode="NEVER",
)
userproxy.initiate_chat(recipient=agent, message="Tell me a joke about openai.")

"""
### Example 1
How to use `AssistantAgent` and `UserProxyAgent` to write code and execute the code.
"""
logger.info("### Example 1")

# %pip install "autogen[retrievechat,lmm]>=0.2.28" -q




http_client = get_openai_httpx_sync_client()  # http_client is needed for openai>1
http_client.__deepcopy__ = types.MethodType(
    lambda self, memo: self, http_client
)  # https://autogenhub.github.io/autogen/docs/topics/llm_configuration#adding-http-client-in-llm_config-for-proxy

config_list = [
    {
        "model": "gpt-4o",
        "http_client": http_client,
        "api_version": "2024-02-01",
        "api_type": "azure",
    },
]

llm_config = {
    "config_list": config_list,
    "temperature": 0,
}


assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # input() doesn't work, so needs to be "NEVER" here
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    },
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)

chat_result = user_proxy.initiate_chat(
    assistant,
    message="""
Who should read this paper: https://arxiv.org/abs/2308.08155
""",
)

logger.debug(f"Cost for the chat:\n{chat_result.cost}")

"""
### Example 2
How to use `AssistantAgent` and `RetrieveUserProxyAgent` to do Retrieval Augmented Generation (RAG) for QA and Code Generation.

Check out this [blog](https://autogenhub.github.io/autogen/blog/2023/10/18/RetrieveChat) for more details.
"""
logger.info("### Example 2")



temp_dir = tempfile.TemporaryDirectory()

code_executor = LocalCommandLineCodeExecutor(
    timeout=40,  # Timeout for each code execution in seconds.
    work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
)


assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    retrieve_config={
        "docs_path": [
            "https://learn.microsoft.com/en-us/fabric/get-started/microsoft-fabric-overview",
            "https://learn.microsoft.com/en-us/fabric/data-science/tuning-automated-machine-learning-visualizations",
        ],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "vector_db": "chroma",  # to use the deprecated `client` parameter, set to None and uncomment the line above
        "overwrite": True,  # set to True if you want to overwrite an existing collection
    },
    code_execution_config={"executor": code_executor},  # Use the local command line code executor.
)

"""
#### 2.1 let's ask a question "List all the Components of Microsoft Fabric".

The answer from **ChatGPT with gpt-4o** at June 7th, 2024 is as below:
```
Microsoft Fabric is a comprehensive data platform that integrates various services and tools for data management, analytics, and collaboration. As of the latest information available, Microsoft Fabric includes the following components:

Data Integration:

Azure Data Factory: For creating, scheduling, and orchestrating data workflows.
Power Query: A data transformation and data preparation tool.
Data Engineering:

Azure Synapse Analytics: For big data and data warehousing solutions, including Synapse SQL, Spark, and Data Explorer.
Data Science:

Azure Machine Learning: For building, training, and deploying machine learning models.
Azure Databricks: For collaborative big data and AI solutions.
Data Warehousing:

...
```

While the answer from AutoGen RAG agent with gpt-4o is as below:
```
The components of Microsoft Fabric are:

1. Power BI
2. Data Factory
3. Data Activator
4. Industry Solutions
5. Real-Time Intelligence
6. Synapse Data Engineering
7. Synapse Data Science
8. Synapse Data Warehouse

Sources: [Microsoft Fabric Overview](https://learn.microsoft.com/en-us/fabric/get-started/microsoft-fabric-overview)
```

AutoGen RAG agent's answer is exactly the right answer per the official documents while ChatGPT made a few mistakes, it even listed Azure Databricks.
"""
logger.info("#### 2.1 let's ask a question "List all the Components of Microsoft Fabric".")

assistant.reset()
problem = "List all the Components of Microsoft Fabric"
chat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=problem)

logger.debug(f"Cost for the chat:\n{chat_result.cost}")

"""
#### 2.2 let's ask it to generate AutoML code for us

The question is "Train a regression model, set time budget to 12s, plot the time line plot after training.".

ChatGPT's answer is as below:

[It showed a figure]

The timeline plot above shows the elapsed time during the training of a linear regression model. The red dashed line indicates the 12-second time budget. The model was trained iteratively, and the plot demonstrates that the training process was monitored to ensure it stayed within the specified time budget.
```

# Create a synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Record the start time
start_time = time.time()

# Train the model and record intermediate times
times = []
time_budget = 12  # in seconds

for _ in range(100):
    model.fit(X_train, y_train)
    current_time = time.time()
    elapsed_time = current_time - start_time
    times.append(elapsed_time)
    if elapsed_time > time_budget:
        break

# Plot the timeline
plt.figure(figsize=(10, 5))
plt.plot(times, label='Training time')
plt.axhline(y=time_budget, color='r', linestyle='--', label='Time Budget (12s)')
plt.xlabel('Iteration')
plt.ylabel('Elapsed Time (s)')
plt.title('Training Time Line Plot')
plt.legend()
plt.grid(True)
plt.show()
```

It's not what I need, as ChatGPT has no context of the [AutoML](https://learn.microsoft.com/en-us/fabric/data-science/tuning-automated-machine-learning-visualizations) solution in Fabric Data Science.

AutoGen RAG agent's answer is much better and ready for deployment. It retrieved the document related to the question and generated code based on the document. It automatically ran the code,  fixed the errors in the code based on the output, and finally it got the correct code.
"""
logger.info("#### 2.2 let's ask it to generate AutoML code for us")

assistant.reset()
problem = "Train a regression model, set time budget to 12s, plot the time line plot after training."

chat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=problem)

logger.debug(f"Cost for the chat:\n{chat_result.cost}")

"""
Below is the code generated by AutoGen RAG agent. It's not a copy of the code in the related document as we asked for different task and training time, but AutoGen RAG agent adapted it very well.
"""
logger.info("Below is the code generated by AutoGen RAG agent. It's not a copy of the code in the related document as we asked for different task and training time, but AutoGen RAG agent adapted it very well.")


housing = fetch_california_housing()
x, y = housing.data, housing.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7654321)

automl = AutoML()
automl_settings = {
    "time_budget": 12,  # Time limit in seconds
    "task": "regression",  # Type of machine learning task
    "log_file_name": "aml_california.log",  # Name of the log file
    "metric": "rmse",  # Evaluation metric
    "log_type": "all",  # Level of logging
}

automl.fit(X_train=x_train, y_train=y_train, **automl_settings)

fig = fviz.plot_timeline(automl)
fig.show()

"""
### Example 3
How to use `MultimodalConversableAgent` to chat with images.

Check out this [blog](https://autogenhub.github.io/autogen/blog/2023/11/06/LMM-Agent) for more details.

We'll ask a question about below image:![image-alt-text](https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0)
"""
logger.info("### Example 3")


image_agent = MultimodalConversableAgent(
    name="image-explainer",
    max_consecutive_auto_reply=10,
    llm_config={"config_list": config_list, "temperature": 0.5, "max_tokens": 300},
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    human_input_mode="NEVER",  # Try between ALWAYS or NEVER
    max_consecutive_auto_reply=0,
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

chat_result = user_proxy.initiate_chat(
    image_agent,
    message="""What's the breed of this dog?
<img https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0>.""",
)

logger.debug(f"Cost for the chat:\n{chat_result.cost}")

logger.info("\n\n[DONE]", bright=True)