from IPython.display import Image, display, Markdown
from datetime import datetime
from google.colab import userdata
from jet.llm.ollama.base_langchain import AzureChatOllama
from jet.logger import CustomLogger
from langchain.output_parsers.ollama_functions import JsonOutputFunctionsParser
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, get_buffer_string
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_ollama_function
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing import Any, Annotated, List
from typing_extensions import TypedDict
import google.generativeai as genai
import operator
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/Alex112525/LangGraph-notebooks/blob/main/Project_Generate_podcast_AI.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Automated Podcast Generation System using LangGraph

### Overview
This notebook demonstrates an automated podcast generation system implemented using LangGraph, Azure Ollama, and Google's Gemini model. The system is designed to generate content for podcasts based on given topics, including keyword generation and structure planning. At the end full podcast will be generated purely based on topic given. Finally extensive (web) based function tool search is also used to augment the needed information for the topic.

### Motivation
Automated content generation systems can significantly reduce the workload for podcast creators while providing structured and relevant content. This implementation showcases how advanced AI models and graph-based workflows can be combined to create a sophisticated system that considers multiple aspects of podcast planning and content creation. Special focus is set on (web) research aspect..

### Key Components
1. State Management: Using TypedDict to define and manage the state of each customer interaction.
2. Query Categorization: Classifying customer queries into Technical, Billing, or General categories.
3. Sentiment Analysis: Determining the emotional tone of customer queries.
4. Response Generation: Creating appropriate responses based on the query category and sentiment.
5. Escalation Mechanism: Automatically escalating queries with negative sentiment to human agents.
6. Workflow Graph: Utilizing LangGraph to create a flexible and extensible workflow.


### Key Classes, Methods, and Functions
#### 1. **State Management**
   - **Class**: `TypedDict` (from `typing_extensions`)
     - Defines and manages the state for each customer interaction.
   - **Classes/Methods**: `MessagesState` (from `langgraph.graph`)
     - Manages messages in the state of the workflow, likely tracking interactions dynamically.

#### 2. **Query Categorization**
   - **Functions/Classes**:
     - `ToolNode` (from `langgraph.prebuilt`)
       - Likely used to create nodes that classify customer queries into categories such as Technical, Billing, or General.
     - Functions within the module may handle classification logic to direct queries to appropriate nodes based on content.

#### 3. **Sentiment Analysis**
   - **Modules/Functions**:
     - The code doesn't specifically list a sentiment analysis function, but it likely leverages LangChain or similar models to determine sentiment through natural language processing (NLP) techniques.
     - You might use tools like `convert_to_ollama_function` (from `langchain_core.utils.function_calling`) to integrate Ollama's models, which could analyze sentiment based on responses.

#### 4. **Response Generation**
   - **Classes/Methods**:
     - `ToolNode` (from `langgraph.prebuilt`) can be configured to generate responses based on query type and sentiment.
     - Functions using LangChain's response parsers, like `JsonOutputFunctionsParser` (from `langchain.output_parsers.ollama_functions`), could help in parsing and generating structured responses.

#### 5. **Escalation Mechanism**
   - **Class/Function**:
     - `tools_condition` (from `langgraph.prebuilt`) could be used to set conditions that trigger escalations based on detected negative sentiment.
     - Escalation logic may also be implemented within the `StateGraph` (from `langgraph.graph`), defining paths that lead to human intervention.

#### 6. **Workflow Graph**
   - **Classes**:
     - `StateGraph` (from `langgraph.graph`)
       - Manages the overall workflow, creating a flexible, extensible path for handling queries and responses.
     - Nodes like `START` and `END` (from `langgraph.graph`) are used to define entry and exit points of the workflow, facilitating a clear process flow.

### Code Key Components

Here we will describe key Classes, Methods and Functions that are involved in the Podcast Agent solution:

Here are the key classes, functions, and methods with numbered points for easy reference:

### **Key Classes**
1. **Planning**: Handles the planning stage of the podcast generation, setting initial parameters and defining strategies for content creation.
2. **Keywords**: Manages the extraction and handling of keywords, crucial for generating focused podcast content.
3. **Subtopics**: Manages the identification and structuring of subtopics related to the main theme of the podcast.
4. **Structure**: Responsible for structuring the overall flow and organization of the podcast content.
5. **InterviewState**: Manages the state of the interview process, keeping track of responses and follow-up questions.
6. **SearchQuery**: Handles the creation and management of search queries for external information retrieval, such as Wikipedia or web search.
7. **ResearchGraphState**: Manages the state graph related to research interactions, potentially integrating with LangGraph for conversational flow management.

### **Key Functions and Methods**
8. **get_model**: Initializes and returns a language model that is used for generating responses or processing content.
9. **get_keywords**: Extracts keywords from given input, using NLP models or APIs to identify relevant terms.
10. **get_structure**: Establishes the structure of the podcast, defining segments, sections, or chapters of the content.
11. **generate_question**: Generates interview questions or prompts based on current context or topic focus.
12. **search_web**: Conducts a web search using APIs or tools integrated in the solution, retrieving results that contribute to the podcast content.
13. **search_wikipedia**: Fetches Wikipedia content to enrich the podcast material, providing background information or supporting details.
14. **generate_answer**: Generates answers to questions posed during the podcast planning or interview phase, using a language model.
15. **save_podcast**: Saves the generated podcast content, exporting it in a specified format.
16. **route_messages**: Routes messages within the conversational state graph, ensuring that content flows logically within the AI-driven conversation.
17. **write_section**: Writes a specific section of the podcast, contributing to structured content creation.
18. **initiate_all_interviews**: Starts all interview processes, managing parallel or sequential interviews for different segments.
19. **write_report**: Writes a report summarizing the generated podcast content or providing insights into the development process.
20. **write_introduction**: Creates the introduction of the podcast, setting the context for the listener.
21. **write_conclusion**: Writes the concluding section of the podcast, wrapping up the main themes and insights.
22. **finalize_report**: Finalizes the report, ensuring that all sections are complete and coherent.
23. **Start_parallel**: Initiates parallel processes, such as handling simultaneous interviews or research queries.

### Method
The system follows a multi-step approach to generate podcast content:

1. [Sub Graph/Agent] Planning: stage is implemented as a subgraph within the larger content generation workflow. This modular approach allows for easy extension and modification of the generation process.
2. [Sub Graph/Agent] Keyword Generation: Identifies at least 5 relevant keywords related to the podcast topic
3. [Sub Graph/Agent] Structure Generation: Creates 5 subtopics based on the podcast topic and 

4. [Main Graph/Agent] Content Generation: Likely generates detailed content for each subtopic
5. [Main Graph/Agent]
Script Formatting: Formats the content into a podcast script
6. [Main Graph/Agent]
Reconciling the individual parts: Reconciling the individual parts (intro, conclusion etc.) into coherent podcast structure.

### Implementation Details

There are two graphs. The first smaller one, is responsible for generating structure, keywords and planning all given a topic. The next main graph takes this information and implements the main logic for creating the podcast. The logic of the main graph (agent) takes these information, produces web search and generates the podcast.

The system uses LangGraph to create a structured workflow for the podcast generation process.
Custom Pydantic models (e.g., Planning, keywords, Subtopics, Structure) are used to ensure type safety and data validation throughout the process.
The notebook sets up necessary API keys and configurations for Azure Ollama, Google Gemini, Tavily (for search), and LangSmith (for monitoring).
The planning subgraph is visualized using a Mermaid diagram, providing a clear representation of the workflow.

### Conclusion
This notebook demonstrates a sophisticated approach to automated podcast content generation by leveraging state-of-the-art AI models and graph-based workflows. The system's modular design allows for easy expansion and customization, making it adaptable to various podcast topics and formats. While the provided code focuses on the planning stage, it lays the groundwork for a comprehensive content generation system that could potentially streamline the podcast creation process.

![Podcast_Gen_LangGraph](../images/podcast_generating_system_langgraph.jpg)

### Import necessary libraries
"""
logger.info("# Automated Podcast Generation System using LangGraph")

pip install langgraph langgraph-sdk langgraph-checkpoint-sqlite langsmith langchain-community langchain-core langchain-ollama tavily-python wikipedia google-generativeai







"""
Make sure to pass the necessary Keys:
"""
logger.info("Make sure to pass the necessary Keys:")

genai.configure(api_key=userdata.get('GEMINI_API_KEY'))
# os.environ["AZURE_OPENAI_API_KEY"] = userdata.get('Azure_ollama')
os.environ["AZURE_OPENAI_ENDPOINT"] = userdata.get('Endpoint_ollama')
os.environ["TAVILY_API_KEY"] = userdata.get('Tavily_API_key')

"""
For langchain as well
"""
logger.info("For langchain as well")

os.environ["LANGCHAIN_API_KEY"] = userdata.get('LangSmith')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "PodcastGenAI"

"""
## Get models

Here we are fetching and configuring the models
"""
logger.info("## Get models")

def get_model(model:str="Agent_test", temp:float=0.1, max_tokens:int=100):
  """Get model from Azure Ollama"""
  model = AzureChatOllama(
        ollama_api_version="2024-02-15-preview",
        azure_deployment=model,
        temperature=temp,
        max_tokens=max_tokens,
    )
  return model

"""
Here we are fetching and configuring the models
"""
logger.info("Here we are fetching and configuring the models")

generation_config = {
  "temperature": 0.21,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 5000,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)
podcast_model = model.start_chat()

"""
## Graphs

### Build Sub-graphs

#### Define State Structure

We define a State class to hold the topic, keywords and subtopics for each topic interaction.

This class definition is crucial for the LangGraph library later.
"""
logger.info("## Graphs")

class Planning(TypedDict):
  topic:str
  keywords: list[str]
  subtopics: list[str]

"""
Here we prompt with following, asking for the keywords relevant for the topic that was provided in the start of the workflow.
"""
logger.info("Here we prompt with following, asking for the keywords relevant for the topic that was provided in the start of the workflow.")

class keywords(BaseModel):
    """Answer with at least 5 keywords that you think are related to the topic"""
    keys: list = Field(description="list of at least 5 keywords related to the topic")

gpt_keywords = get_model("Agent_test",0.1, 50)
model_keywords = gpt_keywords.with_structured_output(keywords)

"""
We will repeat the same process for the Subtopics and Structure. Again important pre requisites in the starter sub-graph
"""
logger.info("We will repeat the same process for the Subtopics and Structure. Again important pre requisites in the starter sub-graph")

class Subtopics(BaseModel):
    """Answer with at least 5 subtopics related to the topic"""
    subtopics: list = Field(description="list of at least 5 subtopics related to the topic")

class Structure(BaseModel):
    """Structure of the podcast having in account the topic and the keywords"""
    subtopics: list[Subtopics] = Field(description="5 subtopics that we will review in the podcast related to the Topic and the Keywords")

gpt_structure = get_model("Agent_test",0.3, 1000)
model_structure = gpt_structure.with_structured_output(Structure)

"""
Here we will pass all of these elements and build the first subgraph.
"""
logger.info("Here we will pass all of these elements and build the first subgraph.")

def get_keywords(state: Planning):
  topic = state['topic']
  messages = [SystemMessage(content="You task is to generate 5 relevant words about the following topic: " + topic)]
  message = model_keywords.invoke(messages)
  return {'keywords': message.keys}

def get_structure(state: Planning):
  topic = state['topic']
  keywords = state['keywords']
  messages = [SystemMessage(content="You task is to generate 5 subtopics to make a podcast about the following topic: " + topic +"and the following keywords:" + " ".join(keywords))]
  message = model_structure.invoke(messages)
  return { "subtopics": message.subtopics[0].subtopics}

plan_builder = StateGraph(Planning)

plan_builder.add_node("Keywords", get_keywords)
plan_builder.add_node("Structure", get_structure)
plan_builder.add_edge(START, "Keywords")
plan_builder.add_edge("Keywords", "Structure")
plan_builder.add_edge("Structure", END)

graph_plan = plan_builder.compile()

display(Image(graph_plan.get_graph(xray=1).draw_mermaid_png()))

graph_plan.invoke({"topic": "What is Attention in human cognition"})

"""
Here we depict what the output looks like in the LangGraph studio:

![agent_1.png](attachment:agent_1.png)

#### Conduct podcast

This is the main part, where the first subbraph is integrated into the main workflow.

Again, we need to define the InterviewState class with all the nodes, that will go into configuring the agent with langgraph
"""
logger.info("#### Conduct podcast")

class InterviewState(MessagesState):
    topic: str # Topic of the podcast
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # Source docs
    section: str # section transcript
    sections: list # Final key we duplicate in outer state for Send() API

"""
Here we define the query to prompt for the interview:
"""
logger.info("Here we define the query to prompt for the interview:")

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

podcast_gpt = get_model(max_tokens= 1000)
structured_llm = podcast_gpt.with_structured_output(SearchQuery)

question_instructions = """You are the host of a popular podcast and you are tasked with interviewing an expert to learn about a specific topic.

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.

2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {topic}
Begin by introducing the topic that fits your goals, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help"

Remember to stay in character throughout your response"""

def generate_question(state: InterviewState):
    """ Node to generate a question """

    topic = state["topic"]
    messages = state["messages"]

    system_message = question_instructions.format(topic=topic)
    question = podcast_gpt.invoke([SystemMessage(content=system_message)]+messages)

    return {"messages": [question]}

"""
Since we are using function calls to parse the web for the particular topics, here we define the search queries for that:
"""
logger.info("Since we are using function calls to parse the web for the particular topics, here we define the search queries for that:")

search_instructions = SystemMessage(content=f"""You will be given a conversation between a host of a popular podcast and an expert.
Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
First, analyze the full conversation.
Pay particular attention to the final question posed by the analyst.
Convert this final question into a well-structured web search query""")

def search_web(state: InterviewState):
    """ Retrieve docs from web search """

    search_query = structured_llm.invoke([search_instructions]+state['messages'])

    tavily_search = TavilySearchResults(max_results = 3)
    search_docs = tavily_search.invoke(search_query.search_query)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def search_wikipedia(state: InterviewState):
    """ Retrieve docs from wikipedia """

    search_query = structured_llm.invoke([search_instructions]+state['messages'])

    search_docs = WikipediaLoader(query=search_query.search_query,
                                  load_max_docs=2).load()

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

answer_instructions = """You are an expert being interviewed by a popular podcast host.
Here is the analyst's focus area: {topic}.
Your goal is to answer a question posed by the interviewer.
To answer the question, use this context:
{context}
When answering questions, follow these steps

1. Use only the information provided in the context.

2. Do not introduce outside information or make assumptions beyond what is explicitly stated in the context.

3. The context includes sources on the topic of each document.

4. Make it interesting."""

def generate_answer(state: InterviewState):

    """ Node to answer a question """

    topic = state["topic"]
    messages = state["messages"]
    context = state["context"]

    system_message = answer_instructions.format(topic=topic, context=context)
    answer = podcast_gpt.invoke([SystemMessage(content=system_message)]+messages)

    answer.name = "expert"

    return {"messages": [answer]}

def save_podcast(state: InterviewState):

    """ save_podcast """

    messages = state["messages"]

    interview = get_buffer_string(messages)

    return {"section": interview}

def route_messages(state: InterviewState, name: str="expert"):
    """ Route between question and answer """

    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    if num_responses >= max_num_turns:
        return 'Save podcast'

    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return 'Save podcast'
    return "Host question"

"""
Final piece of the puzzle are sections that we need to classify the text into:
"""
logger.info("Final piece of the puzzle are sections that we need to classify the text into:")

section_writer_instructions = """You are an expert technical writer.

Your task is to create an interesting, easily digestible section of a podcast based on an interview.

1. Analyze the content of the interview

2. Create a script structure using markdown formatting

3. Make your title engaging based upon the focus area of the analyst:
{focus}

4. For the conversation:
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Mention turns of "Interviewer" and "Expert"
- Aim for approximately 1000 words maximum

5. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

def write_section(state: InterviewState):

    """ Node to answer a question """

    section = state["section"]
    topic = state["topic"]

    system_message = section_writer_instructions.format(focus=topic)
    section_res = podcast_model.send_message(system_message + f"Use this source to write your section: {section}")

    return {"sections": [section_res.text]}

"""
Finally we pass the nodes and the edges to construct our graph:
"""
logger.info("Finally we pass the nodes and the edges to construct our graph:")

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("Host question", generate_question)
interview_builder.add_node("Web research", search_web)
interview_builder.add_node("Wiki research", search_wikipedia)
interview_builder.add_node("Expert answer", generate_answer)
interview_builder.add_node("Save podcast", save_podcast)
interview_builder.add_node("Write script", write_section)

interview_builder.add_edge(START, "Host question")
interview_builder.add_edge("Host question", "Web research")
interview_builder.add_edge("Host question", "Wiki research")
interview_builder.add_edge("Web research", "Expert answer")
interview_builder.add_edge("Wiki research", "Expert answer")
interview_builder.add_conditional_edges("Expert answer", route_messages,['Host question','Save podcast'])
interview_builder.add_edge("Save podcast", "Write script")
interview_builder.add_edge("Write script", END)

memory = MemorySaver()
podcast_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Create podcast")

display(Image(podcast_graph.get_graph().draw_mermaid_png()))

messages = [HumanMessage(f"So you said you were writing an article about Attention in human cognition?")]
thread = {"configurable": {"thread_id": "1"}}
interview = podcast_graph.invoke({"topic": "The Role of Focus in Perception", "messages": messages, "max_num_turns": 2}, thread)
Markdown(interview['sections'][0])

"""
![agent_2.png](attachment:agent_2.png)

### Main graph

In this main graph, we include Research topic, keywords, analysts and all the other elements of the main graph. The end result is the final report that produces the end to end podcast
"""
logger.info("### Main graph")

class ResearchGraphState(TypedDict):
    topic: Annotated[str, operator.add] # Research topic
    keywords: List # Keywords
    max_analysts: int # Number of analysts
    subtopics: List # Subtopics
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report

"""
Prompt for the reporter:
"""
logger.info("Prompt for the reporter:")

report_writer_instructions = """You are a podcast script writer preparing a script for an episode on this overall topic:

{topic}

You have a dedicated researcher who has delved deep into various subtopics related to the main theme.
Your task:

1. You will be given a collection of part of script podcast from the researcher, each covering a different subtopic.
2. Carefully analyze the insights from each script.
3. Consolidate these into a crisp and engaging narrative that ties together the central ideas from all of the script, suitable for a podcast audience.
4. Weave the central points of each script into a cohesive and compelling story, ensuring a natural flow and smooth transitions between segments, creating a unified and insightful exploration of the overall topic.

To format your script:

1. Use markdown formatting.
2. Write in a conversational and engaging tone suitable for a podcast.
3. Seamlessly integrate the insights from each script into the narrative, using clear and concise language.
4. Use transitional phrases and signposting to guide the listener through the different subtopics.

Here are the scripts from the researcher to build your podcast script from:

{context}"""

"""
Prompt for intro
"""
logger.info("Prompt for intro")

intro_instructions = """You are a podcast producer crafting a captivating introduction for an upcoming episode on {topic}.
You will be given an outline of the episode's main segments.
Your job is to write a compelling and engaging introduction that hooks the listener and sets the stage for the discussion.
Include no unnecessary preamble or fluff.
Target around 300 words, using vivid language and intriguing questions to pique the listener's curiosity and preview the key themes and topics covered in the episode.
Use markdown formatting.
Create a catchy and relevant title for the episode and use the # header for the title.
Use ## Introduction as the section header for your introduction.
Here are the segments to draw upon for crafting your introduction: {formatted_str_sections}"""

"""
Prompt for conclusion
"""
logger.info("Prompt for conclusion")

conclusion_instructions = """You are a podcast producer crafting a memorable conclusion for an episode on {topic}.
You will be given an outline of the episode's main segments.
Your job is to write a concise and impactful conclusion that summarizes the key takeaways and leaves a lasting impression on the listener.
Include no unnecessary preamble or fluff.
Target around 200 words, highlighting the most important insights and offering a thought-provoking closing statement that encourages further reflection or action.
Use markdown formatting.
Use ## Conclusion as the section header for your conclusion.
Here are the segments to draw upon for crafting your conclusion: {formatted_str_sections}"""

"""
Core functions that will be parts of the nodes of the graph when passing it to the LangGraph
"""
logger.info("Core functions that will be parts of the nodes of the graph when passing it to the LangGraph")

def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """

    topic = state["topic"]
    return [Send("Create podcast", {"topic": topic,
                                        "messages": [HumanMessage(
                                            content=f"So you said you were researching about {subtopic}?"
                                        )
                                                    ]}) for subtopic in state["subtopics"]]

def write_report(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = podcast_model.send_message(system_message)
    return {"content": report.text}

def write_introduction(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])


    instructions = intro_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    intro = podcast_model.send_message(instructions)
    return {"introduction": intro.text}

def write_conclusion(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])


    instructions = conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    conclusion = podcast_model.send_message(instructions)
    return {"conclusion": conclusion.text}

def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    content = state["content"]
    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]

    return {"final_report": final_report}

def Start_parallel(state):
    """ No-op node that should be interrupted on """
    pass

builder = StateGraph(ResearchGraphState)
builder.add_node("Planing", plan_builder.compile())
builder.add_node("Start research", Start_parallel)
builder.add_node("Create podcast", interview_builder.compile())
builder.add_node("Write report",write_report)
builder.add_node("Write introduction",write_introduction)
builder.add_node("Write conclusion",write_conclusion)
builder.add_node("Finalize podcast",finalize_report)

builder.add_edge(START, "Planing")
builder.add_edge("Planing", "Start research")
builder.add_conditional_edges("Start research", initiate_all_interviews, ["Planing", "Create podcast"])
builder.add_edge("Create podcast", "Write report")
builder.add_edge("Create podcast", "Write introduction")
builder.add_edge("Create podcast", "Write conclusion")
builder.add_edge(["Write introduction", "Write report", "Write conclusion"], "Finalize podcast")
builder.add_edge("Finalize podcast", END)

memory = MemorySaver()
main_graph = builder.compile(checkpointer=memory)
display(Image(main_graph.get_graph(xray=1).draw_mermaid_png()))

"""
Here is an example of asking for a topic regarding attention in human cognition:
"""
logger.info("Here is an example of asking for a topic regarding attention in human cognition:")

topic = "What is Attention in human cognition"

input_g = {"topic":topic}
thread = {"configurable": {"thread_id": "1"}}

for event in main_graph.stream(input_g, thread, stream_mode="updates"):
    logger.debug("--Node--")
    node_name = next(iter(event.keys()))
    logger.debug(node_name)

"""
And here we get the final output:
"""
logger.info("And here we get the final output:")

final_state = main_graph.get_state(thread)
report = final_state.values.get('final_report')
display(Markdown(report))

final_state.values.get('subtopics')

"""
![agent_3.png](attachment:agent_3.png)
"""

logger.info("\n\n[DONE]", bright=True)