from IPython.display import Image, display
from collections.abc import Sequence
from datetime import datetime
from jet.llm.ollama.base_langchain import ChatOllama
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.agents import tool
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from pymongo.mongo_client import MongoClient
from tqdm import tqdm
from typing import Annotated, TypedDict
from typing import Dict, List
import functools
import ollama
import operator
import os
import pandas as pd
import pprint
import random
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# How To Build An Agentic Chatbot With Claude 3.5 Sonnet, LangGraph and MongoDB

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/hr_agentic_chatbot_with_langgraph_claude.ipynb)

## Install Libraries
"""
logger.info("# How To Build An Agentic Chatbot With Claude 3.5 Sonnet, LangGraph and MongoDB")

# !pip install -U --quiet langgraph langchain-community langchain-anthropic langchain-ollama langchain-mongodb langsmith
# !pip install -U --quiet pandas ollama pymongo

"""
## Set Environment Variables
"""
logger.info("## Set Environment Variables")


# os.environ["OPENAI_API_KEY"] = ""
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# os.environ["ANTHROPIC_API_KEY"] = ""
# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

OPEN_AI_EMBEDDING_MODEL = "mxbai-embed-large"
OPEN_AI_EMBEDDING_MODEL_DIMENSION = 256

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "hr_agentic_chatbot"

"""
## Synthetic Data Generation
"""
logger.info("## Synthetic Data Generation")



job_titles = [
    "Software Engineer",
    "Senior Software Engineer",
    "Data Scientist",
    "Product Manager",
    "Project Manager",
    "UX Designer",
    "QA Engineer",
    "DevOps Engineer",
    "CTO",
    "CEO",
]
departments = [
    "IT",
    "Engineering",
    "Data Science",
    "Product",
    "Project Management",
    "Design",
    "Quality Assurance",
    "Operations",
    "Executive",
]

office_locations = [
    "Chicago Office",
    "New York Office",
    "London Office",
    "Berlin Office",
    "Tokyo Office",
    "Sydney Office",
    "Toronto Office",
    "San Francisco Office",
    "Paris Office",
    "Singapore Office",
]


def create_employee(
    employee_id, first_name, last_name, job_title, department, manager_id=None
):
    return {
        "employee_id": employee_id,
        "first_name": first_name,
        "last_name": last_name,
        "gender": random.choice(["Male", "Female"]),
        "date_of_birth": f"{random.randint(1950, 2000)}-{random.randint(1, 12):02}-{random.randint(1, 28):02}",
        "address": {
            "street": f"{random.randint(100, 999)} Main Street",
            "city": "Springfield",
            "state": "IL",
            "postal_code": "62704",
            "country": "USA",
        },
        "contact_details": {
            "email": f"{first_name.lower()}.{last_name.lower()}@example.com",
            "phone_number": f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        },
        "job_details": {
            "job_title": job_title,
            "department": department,
            "hire_date": f"{random.randint(2000, 2022)}-{random.randint(1, 12):02}-{random.randint(1, 28):02}",
            "employment_type": "Full-Time",
            "salary": random.randint(50000, 250000),
            "currency": "USD",
        },
        "work_location": {
            "nearest_office": random.choice(office_locations),
            "is_remote": random.choice([True, False]),
        },
        "reporting_manager": manager_id,
        "skills": random.sample(
            [
                "JavaScript",
                "Python",
                "Node.js",
                "React",
                "Django",
                "Flask",
                "AWS",
                "Docker",
                "Kubernetes",
                "SQL",
            ],
            4,
        ),
        "performance_reviews": [
            {
                "review_date": f"{random.randint(2020, 2023)}-{random.randint(1, 12):02}-{random.randint(1, 28):02}",
                "rating": round(random.uniform(3, 5), 1),
                "comments": random.choice(
                    [
                        "Exceeded expectations in the last project.",
                        "Consistently meets performance standards.",
                        "Needs improvement in time management.",
                        "Outstanding performance and dedication.",
                    ]
                ),
            },
            {
                "review_date": f"{random.randint(2019, 2022)}-{random.randint(1, 12):02}-{random.randint(1, 28):02}",
                "rating": round(random.uniform(3, 5), 1),
                "comments": random.choice(
                    [
                        "Exceeded expectations in the last project.",
                        "Consistently meets performance standards.",
                        "Needs improvement in time management.",
                        "Outstanding performance and dedication.",
                    ]
                ),
            },
        ],
        "benefits": {
            "health_insurance": random.choice(
                ["Gold Plan", "Silver Plan", "Bronze Plan"]
            ),
            "retirement_plan": "401K",
            "paid_time_off": random.randint(15, 30),
        },
        "emergency_contact": {
            "name": f"{random.choice(['Jane', 'Emily', 'Michael', 'Robert'])} {random.choice(['Doe', 'Smith', 'Johnson'])}",
            "relationship": random.choice(["Spouse", "Parent", "Sibling", "Friend"]),
            "phone_number": f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        },
        "notes": random.choice(
            [
                "Promoted to Senior Software Engineer in 2020.",
                "Completed leadership training in 2021.",
                "Received Employee of the Month award in 2022.",
                "Actively involved in company hackathons and innovation challenges.",
            ]
        ),
    }


employees = [
    create_employee("E123456", "John", "Doe", "Software Engineer", "IT", "M987654"),
    create_employee(
        "E123457", "Jane", "Doe", "Senior Software Engineer", "IT", "M987654"
    ),
    create_employee(
        "E123458", "Emily", "Smith", "Data Scientist", "Data Science", "M987655"
    ),
    create_employee(
        "E123459", "Michael", "Brown", "Product Manager", "Product", "M987656"
    ),
    create_employee(
        "E123460", "Sarah", "Davis", "Project Manager", "Project Management", "M987657"
    ),
    create_employee("E123461", "Robert", "Johnson", "UX Designer", "Design", "M987658"),
    create_employee(
        "E123462", "David", "Wilson", "QA Engineer", "Quality Assurance", "M987659"
    ),
    create_employee(
        "E123463", "Chris", "Lee", "DevOps Engineer", "Operations", "M987660"
    ),
    create_employee("E123464", "Sophia", "Garcia", "CTO", "Executive", None),
    create_employee("E123465", "Olivia", "Martinez", "CEO", "Executive", None),
]

df_employees = pd.DataFrame(employees)

csv_file_employees = "synthetic_data_employees.csv"
df_employees.to_csv(csv_file_employees, index=False)

logger.debug(f"Synthetic employee data has been saved to {csv_file_employees}")

df_employees.head()

"""
## Embedding Generation
"""
logger.info("## Embedding Generation")

def create_employee_string(employee):
    job_details = f"{employee['job_details']['job_title']} in {employee['job_details']['department']}"
    skills = ", ".join(employee["skills"])
    performance_reviews = " ".join(
        [
            f"Rated {review['rating']} on {review['review_date']}: {review['comments']}"
            for review in employee["performance_reviews"]
        ]
    )
    basic_info = f"{employee['first_name']} {employee['last_name']}, {employee['gender']}, born on {employee['date_of_birth']}"
    work_location = f"Works at {employee['work_location']['nearest_office']}, Remote: {employee['work_location']['is_remote']}"
    notes = employee["notes"]

    return f"{basic_info}. Job: {job_details}. Skills: {skills}. Reviews: {performance_reviews}. Location: {work_location}. Notes: {notes}"


employee_string = create_employee_string(employees[0])
logger.debug(f"Here's what an employee string looks like: /n {employee_string}")

df_employees["employee_string"] = df_employees.apply(create_employee_string, axis=1)



def get_embedding(text):
    """Generate an embedding for the given text using Ollama's API."""

    if not text or not isinstance(text, str):
        return None

    try:
        embedding = (
            ollama.embeddings.create(
                input=text,
                model=OPEN_AI_EMBEDDING_MODEL,
                dimensions=OPEN_AI_EMBEDDING_MODEL_DIMENSION,
            )
            .data[0]
            .embedding
        )
        return embedding
    except Exception as e:
        logger.debug(f"Error in get_embedding: {e}")
        return None


try:
    df_employees["embedding"] = [
        x
        for x in tqdm(
            df_employees["employee_string"].apply(get_embedding),
            total=len(df_employees),
        )
    ]
    logger.debug("Embeddings generated for employees")
except Exception as e:
    logger.debug(f"Error applying embedding function to DataFrame: {e}")

df_employees.head()

"""
## MongoDB Database Setup

**Steps to creating a MongoDB Database**
- [Register for a free MongoDB Atlas Account](https://www.mongodb.com/cloud/atlas/register?utm_campaign=devrel&utm_source=workshop&utm_medium=organic_social&utm_content=rag%20to%20agents%20notebook&utm_term=richmond.alake)
- [Create a Cluster](https://www.mongodb.com/docs/guides/atlas/cluster/)
- [Get your connection string](https://www.mongodb.com/docs/guides/atlas/connection-string/)
"""
logger.info("## MongoDB Database Setup")

os.environ["MONGO_URI"] = ""

MONGO_URI = os.environ.get("MONGO_URI")


DATABASE_NAME = "demo_company_employees"
COLLECTION_NAME = "employees_records"


def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB and ping the database."""

    client = MongoClient(mongo_uri, appname="devrel.showcase.hr_agent.python")

    try:
        client.admin.command("ping")
        logger.debug("Connection to MongoDB successful")
    except Exception as e:
        logger.debug(f"Error connecting to MongoDB: {e}")
        return None

    return client


if not MONGO_URI:
    logger.debug("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(MONGO_URI)

if mongo_client:
    db = mongo_client.get_database(DATABASE_NAME)
    collection = db.get_collection(COLLECTION_NAME)
else:
    logger.debug("Failed to connect to MongoDB. Exiting...")
    exit(1)

"""
## Data Ingestion
"""
logger.info("## Data Ingestion")

collection.delete_many({})

documents = df_employees.to_dict("records")

collection.insert_many(documents)
logger.debug("Data ingestion into MongoDB completed")

"""
## Vector Search Index Initalisation

1.4 Vector Index Creation

- [Create an Atlas Vector Search Index](https://www.mongodb.com/docs/compass/current/indexes/create-vector-search-index/)

- If you are following this notebook ensure that you are creating a vector search index for the right database(demo_company_employees) and collection(employees_records)

Below is the vector search index definition for this notebook

```json
{
  "fields": [
    {
      "numDimensions": 256,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

- Give your vector search index the name "vector_index" if you are following this notebook

## Agentic System Memory
"""
logger.info("## Vector Search Index Initalisation")



def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        MONGO_URI, session_id, database_name=DATABASE_NAME, collection_name="history"
    )


temp_mem = get_session_history("test")

"""
## LLM Definition
"""
logger.info("## LLM Definition")


llm = ChatOllama(model="llama3.2")

"""
## Tool Definition
"""
logger.info("## Tool Definition")


ATLAS_VECTOR_SEARCH_INDEX = "vector_index"
embedding_model = OllamaEmbeddings(
    model=OPEN_AI_EMBEDDING_MODEL, dimensions=OPEN_AI_EMBEDDING_MODEL_DIMENSION
)

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=DATABASE_NAME + "." + COLLECTION_NAME,
    embedding=embedding_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX,
    text_key="employee_string",
)


@tool
def lookup_employees(query: str, n=10) -> str:
    "Gathers employee details from the database"
    result = vector_store.similarity_search_with_score(query=query, k=n)
    return str(result)


tools = [lookup_employees]

"""
## Agent Definition
"""
logger.info("## Agent Definition")




def create_agent(llm, tools, system_message: str):
    """Create an agent."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}"
                "\nCurrent time: {time}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(time=lambda: str(datetime.now()))
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)

chatbot_agent = create_agent(
    llm,
    tools,
    system_message="You are helpful HR Chabot Agent.",
)

"""
## Node Definition
"""
logger.info("## Node Definition")




def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }


chatbot_node = functools.partial(agent_node, agent=chatbot_agent, name="HR Chatbot")
tool_node = ToolNode(tools, name="tools")

"""
## State Definition
"""
logger.info("## State Definition")




class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

"""
## Agentic Workflow Definition
"""
logger.info("## Agentic Workflow Definition")


workflow = StateGraph(AgentState)

workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("chatbot")
workflow.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})

workflow.add_edge("tools", "chatbot")

"""
## Graph Compiliation and visualisation
"""
logger.info("## Graph Compiliation and visualisation")

graph = workflow.compile()


try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    pass

"""
## Process and View Response
"""
logger.info("## Process and View Response")



events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Build a team to make an iOS app, and tell me the talent gaps"
            )
        ]
    },
    {"recursion_limit": 15},
)


def process_event(event: Dict) -> List[BaseMessage]:
    new_messages = []
    for value in event.values():
        if isinstance(value, dict) and "messages" in value:
            for msg in value["messages"]:
                if isinstance(msg, BaseMessage):
                    new_messages.append(msg)
                elif isinstance(msg, dict) and "content" in msg:
                    new_messages.append(
                        AIMessage(
                            content=msg["content"],
                            additional_kwargs={"sender": msg.get("sender")},
                        )
                    )
                elif isinstance(msg, str):
                    new_messages.append(ToolMessage(content=msg))
    return new_messages


for event in events:
    logger.debug("Event:")
    pprint.plogger.debug(event)
    logger.debug("---")

    new_messages = process_event(event)
    if new_messages:
        temp_mem.add_messages(new_messages)

logger.debug("\nFinal state of temp_mem:")
if hasattr(temp_mem, "messages"):
    for msg in temp_mem.messages:
        logger.debug(f"Type: {msg.__class__.__name__}")
        logger.debug(f"Content: {msg.content}")
        if msg.additional_kwargs:
            logger.debug("Additional kwargs:")
            pprint.plogger.debug(msg.additional_kwargs)
        logger.debug("---")
else:
    logger.debug("temp_mem does not have a 'messages' attribute")

logger.info("\n\n[DONE]", bright=True)