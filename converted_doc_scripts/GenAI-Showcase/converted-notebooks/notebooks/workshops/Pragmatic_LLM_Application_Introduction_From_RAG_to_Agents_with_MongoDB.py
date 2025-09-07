from jet.llm.ollama.base_langchain import ChatOllama, OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMLinguaCompressor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from llmlingua import PromptCompressor
from pymongo.mongo_client import MongoClient
import ollama
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
# **Pragmatic LLM Application Development: From RAG Pipleines to AI Agents**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/workshops/Pragmatic_LLM_Application_Introduction_From_RAG_to_Agents_with_MongoDB.ipynb)

A practical guide that introduces two forms of LLM Applications: RAG (Retrieval-Augmented Generation) pipelines and AI Agents.

This guide is designed to take you on a journey that develops your understanding of LLM Applications, starting with implementations without abstraction frameworks, and later introducing the implementation of RAG pipelines, AI agents, and other LLM application components using frameworks and libraries that alleviate the implementation burden for AI Stack Engineers.

## Key topics covered:

1. **Document Model and MongoDB Integration**: Introduces the Document model and its integration with MongoDB within LLM applications.

2. **RAG Pipeline Fundamentals**: Guides you through the key processes within a RAG pipeline, including data embedding, data ingestion, and handling user queries.

3. **MongoDB Vector Database Integration**: Guides you through the development of a RAG pipeline connected to a MongoDB Vector Database and utilizing Ollama's models.

4. **MongoDB Aggregation Pipelines**: Introduces MongoDB Aggregation pipelines and stages for efficient data retrieval implementation within pipelines.

5. **LLM Abstraction Frameworks**: Showcases the development of RAG pipelines using widely-used LLM abstraction frameworks such as LangChain, LlamaIndex, and HayStack.

6. **Data Handling in LLM Applications**: Presents methods for handling data in LLM applications using tools such as Pydantic and Pandas.

7. **AI Agent Implementation**: Introduces the implementation of AI Agents using libraries such as LangChain and LlamaIndex.

8. **LLM Application Optimization**: Introduces techniques for optimizing LLM Applications, such as prompt compression using the LLMLingua library.

## Who is this for:

- **AI Engineers**: Professionals responsible for developing generative AI applications will find practical guidance on implementing such systems.
- **AI Stack Engineers**: Individuals working with AI Stack tools and libraries will gain insights into the implementation approaches employed by widely adopted libraries, enhancing their understanding and proficiency.
- **Software Engineers**: For those seeking a straightforward introduction to LLM Applications, this guide provides a focused and concise exploration of the subject matter, without unnecessary verbosity or fluff.

# Table of Content

[**Part 1: Vanilla RAG Application**](#scrollTo=hlnz3AIYn5DK)
- [1.1 Synthetic Data Creation](#scrollTo=VXlm_J_TokJp)
- [1.2 Embedding Data for Vector Search](#scrollTo=0AOQw0Caosxu)
- [1.3 Data Ingestion into MongoDB Database](#scrollTo=MhO4jWndsWjR)
- [1.4 Vector Search Index Creation](#scrollTo=B8VZ-c4qt92b)
- [1.5 RAG with MongoDB](#scrollTo=EC6nU1NSuFqO)
- [1.6 Handling User Query](#scrollTo=4UaKjc5nugfd)
- [1.7 Handling User Query With Prompt Compression (LLMLingua)](#scrollTo=BKdB25EMukQO)

[**Part 2: RAG Application With Abstraction Frameworks**](#scrollTo=ALrfaObSteOs)
- [2.1 RAG with LangChain and MongoDB](#scrollTo=DWK6DxuQjmhp)
    - [2.1.3 Prompt Compression with LangChain and LLMLingua](#scrollTo=rnSuWk2cqxtq)
- 2.2 RAG with LlamaIndex and MongoDB
- 2.3 RAG with HayStack and MongoDB

[**Part 3: AI Agent Application: HR Use Case**]()

# Part 1: Vanilla RAG Application

## Install Libaries
"""
logger.info("# **Pragmatic LLM Application Development: From RAG Pipleines to AI Agents**")

# ! pip install pandas ollama pymongo llmlingua

"""
## Set Up Ollama and MongoDB environment variables
"""
logger.info("## Set Up Ollama and MongoDB environment variables")


# os.environ["OPENAI_API_KEY"] = ""

os.environ["MONGO_URI"] = ""


# ollama.api_key = os.environ.get("OPENAI_API_KEY")
OPEN_AI_MODEL = "gpt-4o"
OPEN_AI_EMBEDDING_MODEL = "mxbai-embed-large"
OPEN_AI_EMBEDDING_MODEL_DIMENSION = 1536

"""
## 1.1 Synthetic Data Creation
"""
logger.info("## 1.1 Synthetic Data Creation")



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
## 1.2 Embedding Data For Vector Search
"""
logger.info("## 1.2 Embedding Data For Vector Search")

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
    df_employees["embedding"] = df_employees["employee_string"].apply(get_embedding)
    logger.debug("Embeddings generated for employees")
except Exception as e:
    logger.debug(f"Error applying embedding function to DataFrame: {e}")

df_employees.head()

"""
## 1.3 Data Ingestion into MongoDB Database

**Steps to creating a MongoDB Database**
- [Register for a free MongoDB Atlas Account](https://www.mongodb.com/cloud/atlas/register?utm_campaign=devrel&utm_source=workshop&utm_medium=organic_social&utm_content=rag%20to%20agents%20notebook&utm_term=richmond.alake)
- [Create a Cluster](https://www.mongodb.com/docs/guides/atlas/cluster/)
- [Get your connection string](https://www.mongodb.com/docs/guides/atlas/connection-string/)
"""
logger.info("## 1.3 Data Ingestion into MongoDB Database")

MONGO_URI = os.environ.get("MONGO_URI")

# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

"""
**To be able to connect your notebook to MongoDB Atlas, you need to your IP Access List**
"""

# !curl ifconfig.me


DATABASE_NAME = "demo_company_employees"
COLLECTION_NAME = "employees_records"

def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""

    client = MongoClient(mongo_uri, appname="devrel.showcase.workshop.rag_to_agent")
    logger.debug("Connection to MongoDB successful")
    return client

if not MONGO_URI:
    logger.debug("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(MONGO_URI)

db = mongo_client.get_database(DATABASE_NAME)
collection = db.get_collection(COLLECTION_NAME)

documents = df_employees.to_dict("records")

collection.delete_many({})

collection.insert_many(documents)
logger.debug("Data ingestion into MongoDB completed")

"""
## 1.4 Vector Index Creation

- [Create an Atlas Vector Search Index](https://www.mongodb.com/docs/compass/current/indexes/create-vector-search-index/)

- If you are following this notebook ensure that you are creating a vector search index for the right database(demo_company_employees) and collection(employees_records)

Below is the vector search index definition for this notebook

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

- Give your vector search index the name "vector_index" if you are following this notebook

## 1.5 RAG with MongoDB
"""
logger.info("## 1.4 Vector Index Creation")

def vector_search(user_query, collection, vector_index="vector_index"):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    db (MongoClient.database): The database object.
    collection (MongoCollection): The MongoDB collection to search.
    additional_stages (list): Additional aggregation stages to include in the pipeline.

    Returns:
    list: A list of matching documents.
    """

    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": vector_index,  # specifies the index to use for the search
            "queryVector": query_embedding,  # the vector representing the query
            "path": "embedding",  # field in the documents containing the vectors to search against
            "numCandidates": 150,  # number of candidate matches to consider
            "limit": 5,  # return top 20 matches
        }
    }

    pipeline = [vector_search_stage]

    results = collection.aggregate(pipeline)

    return list(results)

"""
## 1.6 Handling User Query
"""
logger.info("## 1.6 Handling User Query")

def handle_user_query(query, collection):
    get_knowledge = vector_search(query, collection)

    search_result = ""

    for result in get_knowledge:
        reporting_manager = result.get("reporting_manager")
        if isinstance(reporting_manager, dict):
            manager_id = reporting_manager.get("manager_id", "N/A")
        else:
            manager_id = "N/A"

        employee_profile = f"""
      Employee ID: {result.get('employee_id', 'N/A')}
      Name: {result.get('first_name', 'N/A')} {result.get('last_name', 'N/A')}
      Gender: {result.get('gender', 'N/A')}
      Date of Birth: {result.get('date_of_birth', 'N/A')}
      Address: {result.get('address', {}).get('street', 'N/A')}, {result.get('address', {}).get('city', 'N/A')}, {result.get('address', {}).get('state', 'N/A')}, {result.get('address', {}).get('postal_code', 'N/A')}, {result.get('address', {}).get('country', 'N/A')}
      Contact Details: Email - {result.get('contact_details', {}).get('email', 'N/A')}, Phone - {result.get('contact_details', {}).get('phone_number', 'N/A')}
      Job Details: Title - {result.get('job_details', {}).get('job_title', 'N/A')}, Department - {result.get('job_details', {}).get('department', 'N/A')}, Hire Date - {result.get('job_details', {}).get('hire_date', 'N/A')}, Type - {result.get('job_details', {}).get('employment_type', 'N/A')}, Salary - {result.get('job_details', {}).get('salary', 'N/A')} {result.get('job_details', {}).get('currency', 'N/A')}
      Work Location: Nearest Office - {result.get('work_location', {}).get('nearest_office', 'N/A')}, Remote - {result.get('work_location', {}).get('is_remote', 'N/A')}
      Reporting Manager: ID - {manager_id}
      Skills: {', '.join(result.get('skills', ['N/A']))}
      Performance Reviews: {', '.join([f"Date: {review.get('review_date', 'N/A')}, Rating: {review.get('rating', 'N/A')}, Comments: {review.get('comments', 'N/A')}" for review in result.get('performance_reviews', [])])}
      Benefits: Health Insurance - {result.get('benefits', {}).get('health_insurance', 'N/A')}, Retirement Plan - {result.get('benefits', {}).get('retirement_plan', 'N/A')}, PTO - {result.get('benefits', {}).get('paid_time_off', 'N/A')} days
      Emergency Contact: Name - {result.get('emergency_contact', {}).get('name', 'N/A')}, Relationship - {result.get('emergency_contact', {}).get('relationship', 'N/A')}, Phone - {result.get('emergency_contact', {}).get('phone_number', 'N/A')}
      Notes: {result.get('notes', 'N/A')}
      """
        search_result += employee_profile + "\n"

    prompt = (
        "Answer this user query: "
        + query
        + " with the following context: "
        + search_result
    )
    logger.debug("Uncompressed Prompt:\n")
    logger.debug(prompt)

    completion = ollama.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an Human Resource System within a corporate company.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return (completion.choices[0].message.content), search_result

query = "Who is the CEO?"
response, source_information = handle_user_query(query, collection)

logger.debug(f"Response: {response}")

"""
## 1.7 Handling User Query (With Prompt Compression)
"""
logger.info("## 1.7 Handling User Query (With Prompt Compression)")




llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    model_config={"revision": "main"},
    use_llmlingua2=True,
    device_map="cpu",  # change to 'cuda' if gpu is availabe on device
)


def compress_query_prompt(context):
    compressed_prompt = llm_lingua.compress_prompt(
        str(context),
        rate=0.33,
        force_tokens=["!", ".", "?", "\n"],
        drop_consecutive=True,
    )

    logger.debug("------")
    logger.debug(compressed_prompt)
    logger.debug("-------")

    return compressed_prompt



def handle_user_query_with_compression(query, collection):
    get_knowledge = vector_search(query, collection)

    search_result = ""

    for result in get_knowledge:
        employee_profile = f"""
      Employee ID: {result.get('employee_id', 'N/A')}
      Name: {result.get('first_name', 'N/A')} {result.get('last_name', 'N/A')}
      Gender: {result.get('gender', 'N/A')}
      Date of Birth: {result.get('date_of_birth', 'N/A')}
      Address: {result.get('address', {}).get('street', 'N/A')}, {result.get('address', {}).get('city', 'N/A')}, {result.get('address', {}).get('state', 'N/A')}, {result.get('address', {}).get('postal_code', 'N/A')}, {result.get('address', {}).get('country', 'N/A')}
      Contact Details: Email - {result.get('contact_details', {}).get('email', 'N/A')}, Phone - {result.get('contact_details', {}).get('phone_number', 'N/A')}
      Job Details: Title - {result.get('job_details', {}).get('job_title', 'N/A')}, Department - {result.get('job_details', {}).get('department', 'N/A')}, Hire Date - {result.get('job_details', {}).get('hire_date', 'N/A')}, Type - {result.get('job_details', {}).get('employment_type', 'N/A')}, Salary - {result.get('job_details', {}).get('salary', 'N/A')} {result.get('job_details', {}).get('currency', 'N/A')}
      Work Location: Nearest Office - {result.get('work_location', {}).get('nearest_office', 'N/A')}, Remote - {result.get('work_location', {}).get('is_remote', 'N/A')}
      Skills: {', '.join(result.get('skills', ['N/A']))}
      Performance Reviews: {', '.join([f"Date: {review.get('review_date', 'N/A')}, Rating: {review.get('rating', 'N/A')}, Comments: {review.get('comments', 'N/A')}" for review in result.get('performance_reviews', [])])}
      Benefits: Health Insurance - {result.get('benefits', {}).get('health_insurance', 'N/A')}, Retirement Plan - {result.get('benefits', {}).get('retirement_plan', 'N/A')}, PTO - {result.get('benefits', {}).get('paid_time_off', 'N/A')} days
      Emergency Contact: Name - {result.get('emergency_contact', {}).get('name', 'N/A')}, Relationship - {result.get('emergency_contact', {}).get('relationship', 'N/A')}, Phone - {result.get('emergency_contact', {}).get('phone_number', 'N/A')}
      Notes: {result.get('notes', 'N/A')}
      """
        search_result += employee_profile + "\n"

    query_info = {
        "demonstration_str": search_result,  # Results from information retrieval process
        "instruction": "Write a high-quality answer for the given question using only the provided search results.",
        "question": query,
    }

    compressed_prompt = compress_query_prompt(query_info)

    prompt = f"Answer this user query: {query} with the following context:\n{compressed_prompt}"
    logger.debug("Compressed Prompt:\n")
    pprint.plogger.debug(prompt)

    completion = ollama.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an Human Resource System within a corporate company.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return (completion.choices[0].message.content), search_result

query = "Who is the CEO?"
response, source_information = handle_user_query_with_compression(query, collection)

logger.debug(f"Response: {response}")

"""
# Part 2: RAG Application: HR Use Case (POLM AI Stack)

### RAG with Langchain and MongoDB
"""
logger.info("# Part 2: RAG Application: HR Use Case (POLM AI Stack)")

# !pip install --upgrade --quiet langchain langchain-mongodb langchain-ollama langchain_community pymongo


embedding_model = OllamaEmbeddings(
    model=OPEN_AI_EMBEDDING_MODEL, dimensions=OPEN_AI_EMBEDDING_MODEL_DIMENSION
)

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=DATABASE_NAME + "." + COLLECTION_NAME,
    embedding=embedding_model,
    index_name="vector_index",
    text_key="employee_string",
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)

llm = ChatOllama(model="llama3.2")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)
question = "Who is the CEO??"
answer = rag_chain.invoke(question)

logger.debug("Question: " + question)
logger.debug("Answer: " + answer)

"""
#### Prompt Compression with LangChain and LLMLingua
"""
logger.info("#### Prompt Compression with LangChain and LLMLingua")


compressor = LLMLinguaCompressor(model_name="ollama-community/gpt2", device_map="cpu")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke("Who is the CEO?")
logger.debug(compressed_docs)


chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)

chain.invoke({"query": "Who is the CEO?"})

"""
### RAG with LlamaIndex and MongoDB (Coming Soon)

### RAG with HayStack and MongoDB (Coming Soon)

# Part 3: AI Agent Application: HR Use Case (POLM AI Stack)

### AI Agents with langChain and MongoDB (Coming Soon)

### AI Agents with LlamaIndex and MongoDB (Coming Soon)

### AI Agents with HayStack and MongoDB (Coming Soon)
"""
logger.info("### RAG with LlamaIndex and MongoDB (Coming Soon)")


logger.info("\n\n[DONE]", bright=True)