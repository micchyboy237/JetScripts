from autogen.agentchat.contrib.agent_builder import AgentBuilder
from jet.logger import CustomLogger
import autogen
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# AutoBuild
By: [Linxin Song](https://linxins97.github.io/), [Jieyu Zhang](https://jieyuz2.github.io/)
Reference: [Agent AutoBuild](https://autogenhub.github.io/autogen/blog/2023/11/26/Agent-AutoBuild/)

AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

In this notebook, we introduce a new class, `AgentBuilder`, to help user build an automatic task solving process powered by multi-agent system. Specifically, in `build()`, we prompt a LLM to create multiple participant agent and initialize a group chat, and specify whether this task need programming to solve. AgentBuilder also support open-source LLMs by [vLLM](https://docs.vllm.ai/en/latest/index.html) and [Fastchat](https://github.com/lm-sys/FastChat). Check the supported model list [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Requirement

AutoBuild require `autogen[autobuild]`, which can be installed by the following command:
"""
logger.info("# AutoBuild")

# %pip install autogen[autobuild]

"""
## Step 1: prepare configuration and some useful functions
Prepare a `config_file_or_env` for assistant agent to limit the choice of LLM you want to use in this task. This config can be a path of json file or a name of environment variable. A `default_llm_config` is also required for initialize the specific config of LLMs like seed, temperature, etc...
"""
logger.info("## Step 1: prepare configuration and some useful functions")


config_file_or_env = "OAI_CONFIG_LIST"
llm_config = {"temperature": 0}
config_list = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-4-turbo", "gpt-4"]})


def start_task(execution_task: str, agent_list: list, coding=True):
    group_chat = autogen.GroupChat(
        agents=agent_list,
        messages=[],
        max_round=12,
        allow_repeat_speaker=agent_list[:-1] if coding is True else agent_list,
    )
    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": config_list, **llm_config},
    )
    agent_list[0].initiate_chat(manager, message=execution_task)

"""
## Step 2: create a AgentBuilder
Create a `AgentBuilder` with the specified `config_path_or_env`. AgentBuilder will use `gpt-4` in default to complete the whole process, you can specify the `builder_model` and `agent_model` to other Ollama model to match your task. 
You can also specify an open-source LLM supporting by vLLM and FastChat, see blog for more details.
"""
logger.info("## Step 2: create a AgentBuilder")

builder = AgentBuilder(
    config_file_or_env=config_file_or_env, builder_model=["gpt-4-turbo"], agent_model=["gpt-4-turbo"]
)

"""
## Step 3: specify a building task

Specify a building task with a general description. Building task will help build manager (a LLM) decide what agents should be built.
"""
logger.info("## Step 3: specify a building task")

building_task = "Generate some agents that can find papers on arxiv by programming and analyzing them in specific domains related to computer science and medical science."

"""
## Step 4: build group chat agents
Use `build()` to let build manager (the specified `builder_model`) complete the group chat agents generation. If you think coding is necessary in your task, you can use `coding=True` to add a user proxy (an automatic code interpreter) into the agent list, like: 
```python
builder.build(building_task, default_llm_config, coding=True)
```
If `coding` is not specified, AgentBuilder will determine on its own whether the user proxy should be added or not according to the task.
"""
logger.info("## Step 4: build group chat agents")

agent_list, agent_configs = builder.build(building_task, llm_config)

"""
## Step 5: execute task
Let agents generated in `build()` to complete the task collaboratively in a group chat.
"""
logger.info("## Step 5: execute task")

start_task(
    execution_task="Find a recent paper about gpt-4 on arxiv and find its potential applications in software.",
    agent_list=agent_list,
    coding=agent_configs["coding"],
)

"""
## Step 6 (Optional): clear all agents and prepare for the next task
You can clear all agents generated in this task by the following code if your task is completed or the next task is largely different from the current task. If the agent's backbone is an open-source LLM, this process will also shut down the endpoint server. If necessary, you can use `recycle_endpoint=False` to retain the previous open-source LLMs' endpoint server.
"""
logger.info("## Step 6 (Optional): clear all agents and prepare for the next task")

builder.clear_all_agents(recycle_endpoint=True)

"""
## Save & load configs

You can save all necessary information of the built group chat agents. Here is a case for those agents generated in the above task:
```json
{
    "building_task": "Generate some agents that can find papers on arxiv by programming and analyzing them in specific domains related to computer science and medical science.",
    "agent_configs": [
        {
            "name": "ArXiv_Data_Scraper_Developer",
            "model": "gpt-4-1106-preview",
            "system_message": "You are now in a group chat. You need to complete a task with other participants. As an ArXiv_Data_Scraper_Developer, your focus is to create and refine tools capable of intelligent search and data extraction from arXiv, honing in on topics within the realms of computer science and medical science. Utilize your proficiency in Python programming to design scripts that navigate, query, and parse information from the platform, generating valuable insights and datasets for analysis. \n\nDuring your mission, it\u2019s not just about formulating queries; your role encompasses the optimization and precision of the data retrieval process, ensuring relevance and accuracy of the information extracted. If you encounter an issue with a script or a discrepancy in the expected output, you are encouraged to troubleshoot and offer revisions to the code you find in the group chat.\n\nWhen you reach a point where the existing codebase does not fulfill task requirements or if the operation of provided code is unclear, you should ask for help from the group chat manager. They will facilitate your advancement by providing guidance or appointing another participant to assist you. Your ability to adapt and enhance scripts based on peer feedback is critical, as the dynamic nature of data scraping demands ongoing refinement of techniques and approaches.\n\nWrap up your participation by confirming the user's need has been satisfied with the data scraping solutions you've provided. Indicate the completion of your task by replying \"TERMINATE\" in the group chat.",
            "description": "ArXiv_Data_Scraper_Developer is a specialized software development role requiring proficiency in Python, including familiarity with web scraping libraries such as BeautifulSoup or Scrapy, and a solid understanding of APIs and data parsing. They must possess the ability to identify and correct errors in existing scripts and confidently engage in technical discussions to improve data retrieval processes. The role also involves a critical eye for troubleshooting and optimizing code to ensure efficient data extraction from the ArXiv platform for research and analysis purposes."
        },
        {
            "name": "Computer_Science_Research_Analyst",
            "model": "gpt-4-1106-preview",
            "system_message": "You are now in a group chat. You need to complete a task with other participants. As a Computer Science Research Analyst, your objective is to utilize your analytical capabilities to identify and examine scholarly articles on arXiv, focusing on areas bridging computer science and medical science. Employ Python for automation where appropriate and leverage your expertise in the subject matter to draw insights from the research.\n\nEnsure that the information is acquired systematically; tap into online databases, interpret data sets, and perform literature reviews to pinpoint relevant findings. Should you encounter a complex problem or if you find your progress stalled, feel free to question the existing approaches discussed in the chat or contribute an improved method or analysis.\n\nIf the task proves to be beyond your current means or if you face uncertainty at any stage, seek assistance from the group chat manager. The manager is available to provide guidance or to involve another expert if necessary to move forward effectively.\n\nYour contributions are crucial, and it is important to communicate your findings and conclusions clearly. Once you believe the task is complete and the group's need has been satisfied, please affirm the completion by replying \"TERMINATE\".",
            "description": "Computer_Science_Research_Analyst is a role requiring strong analytical skills, a deep understanding of computer science concepts, and proficiency in Python for data analysis and automation. This position should have the ability to critically assess the validity of information, challenge assumptions, and provide evidence-based corrections or alternatives. They should also have excellent communication skills to articulate their findings and suggestions effectively within the group chat."
        },
        {
            "name": "Medical_Science_Research_Analyst",
            "model": "gpt-4-1106-preview",
            "system_message": "You are now in a group chat. You need to complete a task with other participants. As a Medical_Science_Research_Analyst, your function is to harness your analytical strengths and understanding of medical research to source and evaluate pertinent papers from the arXiv database, focusing on the intersection of computer science and medical science. Utilize your Python programming skills to automate data retrieval and analysis tasks. Engage in systematic data mining to extract relevant content, then apply your analytical expertise to interpret the findings qualitatively. \n\nWhen there is a requirement to gather information, employ Python scripts to automate the aggregation process. This could include scraping web data, retrieving and processing documents, and performing content analyses. When these scripts produce outputs, use your subject matter expertise to evaluate the results. \n\nProgress through your task step by step. When an explicit plan is absent, present a structured outline of your intended methodology. Clarify which segments of the task are handled through automation, and which necessitate your interpretative skills. \n\nIn the event code is utilized, the script type must be specified. You are expected to execute the scripts provided without making changes. Scripts are to be complete and functionally standalone. Should you encounter an error upon execution, critically review the output, and if needed, present a revised script for the task at hand. \n\nFor tasks that require saving and executing scripts, indicate the intended filename at the beginning of the script. \n\nMaintain clear communication of the results by harnessing the 'print' function where applicable. If an error arises or a task remains unsolved after successful code execution, regroup to collect additional information, reassess your approach, and explore alternative strategies. \n\nUpon reaching a conclusion, substantiate your findings with credible evidence where possible.\n\nConclude your participation by confirming the task's completion with a \"TERMINATE\" response.\n\nShould uncertainty arise at any point, seek guidance from the group chat manager for further directives or reassignment of the task.",
            "description": "The Medical Science Research Analyst is a professionally trained individual with strong analytical skills, specializing in interpreting and evaluating scientific research within the medical field. They should possess expertise in data analysis, likely with proficiency in Python for analyzing datasets, and have the ability to critically assess the validity and relevance of previous messages or findings relayed in the group chat. This role requires a solid foundation in medical knowledge to provide accurate and evidence-based corrections or insights."
        },
        {
            "name": "Data_Analysis_Engineer",
            "model": "gpt-4-1106-preview",
            "system_message": "You are now in a group chat. You need to complete a task with other participants. As a Data Analysis Engineer, your role involves leveraging your analytical skills to gather, process, and analyze large datasets. You will employ various data analysis techniques and tools, particularly Python for scripting, to extract insights from the data related to computer science and medical science domains on arxiv.\n\nIn scenarios where information needs to be collected or analyzed, you will develop Python scripts to automate the data retrieval and processing tasks. For example, you may write scripts to scrape the arXiv website, parse metadata of research papers, filter content based on specific criteria, and perform statistical analysis or data visualization. \n\nYour workflow will include the following steps:\n\n1. Use your Python coding abilities to design scripts for data extraction and analysis. This can involve browsing or searching the web, downloading and reading files, or printing the content of web pages or files relevant to the given domains.\n2. After gathering the necessary data, apply your data analysis expertise to derive meaningful insights or patterns present in the data. This should be done methodically, making the most of your Python skills for data manipulation and interpretation.\n3. Communicate your findings clearly to the group chat. Ensure the results are straightforward for others to understand and act upon.\n4. If any issues arise from executing the code, such as lack of output or unexpected results, you can question the previous messages or code in the group chat and attempt to provide a corrected script or analysis.\n5. When uncertain or facing a complex problem that you cannot solve alone, ask for assistance from the group chat manager. They can either provide guidance or assign another participant to help you.\n\nOnce you believe the task is completed satisfactorily, and you have fulfilled the user's need, respond with \"TERMINATE\" to signify the end of your contribution to the task. Remember, while technical proficiency in Python is essential for this role, the ability to work collaboratively within the group chat, communicate effectively, and adapt to challenges is equally important.",
            "description": "Data_Analysis_Engineer is a professional adept in collecting, analyzing, and interpreting large datasets, using statistical tools and machine learning techniques to provide actionable insights. They should possess strong Python coding skills for data manipulation and analysis, an understanding of database management, as well as the ability to communicate complex results effectively to non-technical stakeholders. This position should be allowed to speak when data-driven clarity is needed or when existing analyses or methodologies are called into question."
        },
        {
            "name": "ML_Paper_Summarization_Specialist",
            "model": "gpt-4-1106-preview",
            "system_message": "You are now in a group chat. You need to complete a task with other participants. As an ML_Paper_Summarization_Specialist, your role entails leveraging machine learning techniques to extract and analyze academic papers from arXiv, focusing on domains that intersect computer science and medical science. Utilize your expertise in natural language processing and data analysis to identify relevant papers, extract key insights, and generate summaries that accurately reflect the advancements and findings within those papers.\n\nYou are expected to apply your deep understanding of machine learning algorithms, data mining, and information retrieval to construct models and systems that can efficiently process and interpret scientific literature.\n\nIf you encounter any challenges in accessing papers, parsing content, or algorithmic processing, you may seek assistance by presenting your issue to the group chat. Should there be a disagreement regarding the efficacy of a method or the accuracy of a summarization, you are encouraged to critically evaluate previous messages or outputs and offer improved solutions to enhance the group's task performance.\n\nShould confusion arise during the task, rather than relying on coding scripts, please request guidance from the group chat manager, and allow them to facilitate the necessary support by inviting another participant who can aid in overcoming the current obstacle.\n\nRemember, your primary duty is to synthesize complex academic content into concise, accessible summaries that will serve as a valuable resource for researchers and professionals seeking to stay abreast of the latest developments in their respective fields. \n\nOnce you believe your task is completed and the summaries provided meet the necessary standards of accuracy and comprehensiveness, reply \"TERMINATE\" to signal the end of your contribution to the group's task.",
            "description": "The ML_Paper_Summarization_Specialist is a professional adept in machine learning concepts and current research trends, with strong analytical skills to critically evaluate information, synthesizing knowledge from academic papers into digestible summaries. This specialist should be proficient in Python for text processing and have the ability to provide constructive feedback on technical discussions, guide effective implementation, and correct misconceptions or errors related to machine learning theory and practice in the chat. They should be a reliable resource for clarifying complex information and ensuring accurate application of machine learning techniques within the group chat context."
        }
    ],
    "coding": true,
    "default_llm_config": {
        "temperature": 0
    },
    "code_execution_config": {
        "work_dir": "groupchat",
        "use_docker": false,
        "timeout": 60,
        "last_n_messages": 2
    }
}
```
These information will be saved in JSON format. You can provide a specific filename, otherwise, AgentBuilder will save config to the current path with a generated filename 'save_config_TASK_MD5.json'.
"""
logger.info("## Save & load configs")

saved_path = builder.save()

"""
After that, you can load the saved config and skip the building process. AgentBuilder will create agents with those information without prompting the builder manager.
"""
logger.info("After that, you can load the saved config and skip the building process. AgentBuilder will create agents with those information without prompting the builder manager.")

new_builder = AgentBuilder(config_file_or_env=config_file_or_env)
agent_list, agent_configs = new_builder.load(
    "./save_config_c52224ebd16a2e60b348f3f04ac15e79.json"
)  # load previous agent configs
start_task(
    execution_task="Find a recent paper about LLaVA on arxiv and find its potential applications in computer vision.",
    agent_list=agent_list,
)
new_builder.clear_all_agents()

logger.info("\n\n[DONE]", bright=True)