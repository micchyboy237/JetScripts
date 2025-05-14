from dotenv import load_dotenv
from jet.logger import CustomLogger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from typing import List
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/proposition_chunking.ipynb)

# Propositions Chunking

### Overview

This code implements the proposition chunking method, based on [research from Tony Chen, et. al.](https://doi.org/10.48550/arXiv.2312.06648). The system break downs the input text into propositions that are atomic, factual, self-contained, and concise in nature, encodes the propositions into a vectorstore, which can be later used for retrieval

### Key Components

1. **Document Chunking:** Splitting a document into manageable pieces for analysis.
2. **Proposition Generation:** Using LLMs to break down document chunks into factual, self-contained propositions.
3. **Proposition Quality Check:** Evaluating generated propositions based on accuracy, clarity, completeness, and conciseness.
4. **Embedding and Vector Store:** Embedding both propositions and larger chunks of the document into a vector store for efficient retrieval.
5. **Retrieval and Comparison:** Testing the retrieval system with different query sizes and comparing results from the proposition-based model with the larger chunk-based model.

<img src="../images/proposition_chunking.svg" alt="Reliable-RAG" width="600">

### Motivation

The motivation behind the propositions chunking method is to build a system that breaks down a text document into concise, factual propositions for more granular and precise information retrieval. Using propositions allows for finer control and better handling of specific queries, particularly for extracting knowledge from detailed or complex texts. The comparison between using smaller proposition chunks and larger document chunks aims to evaluate the effectiveness of granular information retrieval.

### Method Details

1. **Loading Environment Variables:** The code begins by loading environment variables (e.g., API keys for the LLM service) to ensure that the system can access the necessary resources.
   
2. **Document Chunking:**
   - The input document is split into smaller pieces (chunks) using `RecursiveCharacterTextSplitter`. This ensures that each chunk is of manageable size for the LLM to process.
   
3. **Proposition Generation:**
   - Propositions are generated from each chunk using an LLM (in this case, "llama-3.1-70b-versatile"). The output is structured as a list of factual, self-contained statements that can be understood without additional context.
   
4. **Quality Check:**
   - A second LLM evaluates the quality of the propositions by scoring them on accuracy, clarity, completeness, and conciseness. Propositions that meet the required thresholds in all categories are retained.
   
5. **Embedding Propositions:**
   - Propositions that pass the quality check are embedded into a vector store using the `OllamaEmbeddings` model. This allows for similarity-based retrieval of propositions when queries are made.
   
6. **Retrieval and Comparison:**
   - Two retrieval systems are built: one using the proposition-based chunks and another using larger document chunks. Both are tested with several queries to compare their performance and the precision of the returned results.

### Benefits

- **Granularity:** By breaking the document into small factual propositions, the system allows for highly specific retrieval, making it easier to extract precise answers from large or complex documents.
- **Quality Assurance:** The use of a quality-checking LLM ensures that the generated propositions meet specific standards, improving the reliability of the retrieved information.
- **Flexibility in Retrieval:** The comparison between proposition-based and larger chunk-based retrieval allows for evaluating the trade-offs between granularity and broader context in search results.

### Implementation

1. **Proposition Generation:** The LLM is used in conjunction with a custom prompt to generate factual statements from the document chunks.
2. **Quality Checking:** The generated propositions are passed through a grading system that evaluates accuracy, clarity, completeness, and conciseness.
3. **Vector Store Integration:** Propositions are stored in a FAISS vector store after being embedded using a pre-trained embedding model, allowing for efficient similarity-based search and retrieval.
4. **Query Testing:** Multiple test queries are made to the vector stores (proposition-based and larger chunks) to compare the retrieval performance.

### Summary

This code presents a robust method for breaking down a document into self-contained propositions using LLMs. The system performs a quality check on each proposition, embeds them in a vector store, and retrieves the most relevant information based on user queries. The ability to compare granular propositions against larger document chunks provides insight into which method yields more accurate or useful results for different types of queries. The approach emphasizes the importance of high-quality proposition generation and retrieval for precise information extraction from complex documents.

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Propositions Chunking")

# !pip install faiss-cpu langchain langchain-community python-dotenv


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY') # For LLM

"""
### Test Document
"""
logger.info("### Test Document")

sample_content = """Paul Graham's essay "Founder Mode," published in September 2024, challenges conventional wisdom about scaling startups, arguing that founders should maintain their unique management style rather than adopting traditional corporate practices as their companies grow.
Conventional Wisdom vs. Founder Mode
The essay argues that the traditional advice given to growing companies—hiring good people and giving them autonomy—often fails when applied to startups.
This approach, suitable for established companies, can be detrimental to startups where the founder's vision and direct involvement are crucial. "Founder Mode" is presented as an emerging paradigm that is not yet fully understood or documented, contrasting with the conventional "manager mode" often advised by business schools and professional managers.
Unique Founder Abilities
Founders possess unique insights and abilities that professional managers do not, primarily because they have a deep understanding of their company's vision and culture.
Graham suggests that founders should leverage these strengths rather than conform to traditional managerial practices. "Founder Mode" is an emerging paradigm that is not yet fully understood or documented, with Graham hoping that over time, it will become as well-understood as the traditional manager mode, allowing founders to maintain their unique approach even as their companies scale.
Challenges of Scaling Startups
As startups grow, there is a common belief that they must transition to a more structured managerial approach. However, many founders have found this transition problematic, as it often leads to a loss of the innovative and agile spirit that drove the startup's initial success.
Brian Chesky, co-founder of Airbnb, shared his experience of being advised to run the company in a traditional managerial style, which led to poor outcomes. He eventually found success by adopting a different approach, influenced by how Steve Jobs managed Apple.
Steve Jobs' Management Style
Steve Jobs' management approach at Apple served as inspiration for Brian Chesky's "Founder Mode" at Airbnb. One notable practice was Jobs' annual retreat for the 100 most important people at Apple, regardless of their position on the organizational chart
. This unconventional method allowed Jobs to maintain a startup-like environment even as Apple grew, fostering innovation and direct communication across hierarchical levels. Such practices emphasize the importance of founders staying deeply involved in their companies' operations, challenging the traditional notion of delegating responsibilities to professional managers as companies scale.
"""

"""
### Chunking
"""
logger.info("### Chunking")


embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

docs_list = [Document(page_content=sample_content, metadata={"Title": "Paul Graham's Founder Mode Essay", "Source": "https://www.perplexity.ai/page/paul-graham-s-founder-mode-ess-t9TCyvkqRiyMQJWsHr0fnQ"})]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)

for i, doc in enumerate(doc_splits):
    doc.metadata['chunk_id'] = i+1 ### adding chunk id

"""
### Generate Propositions
"""
logger.info("### Generate Propositions")


class GeneratePropositions(BaseModel):
    """List of all the propositions in a given document"""

    propositions: List[str] = Field(
        description="List of propositions (factual, self-contained, and concise information)"
    )


llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
structured_llm= llm.with_structured_output(GeneratePropositions)

proposition_examples = [
    {"document":
        "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.",
     "propositions":
        "['Neil Armstrong was an astronaut.', 'Neil Armstrong walked on the Moon in 1969.', 'Neil Armstrong was the first person to walk on the Moon.', 'Neil Armstrong walked on the Moon during the Apollo 11 mission.', 'The Apollo 11 mission occurred in 1969.']"
    },
]

example_proposition_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{document}"),
        ("ai", "{propositions}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_proposition_prompt,
    examples = proposition_examples,
)

system = """Please break down the following text into simple, self-contained propositions. Ensure that each proposition meets the following criteria:

    1. Express a Single Fact: Each proposition should state one specific fact or claim.
    2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
    3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
    4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
    5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        few_shot_prompt,
        ("human", "{document}"),
    ]
)

proposition_generator = prompt | structured_llm

propositions = [] # Store all the propositions from the document

for i in range(len(doc_splits)):
    response = proposition_generator.invoke({"document": doc_splits[i].page_content}) # Creating proposition
    for proposition in response.propositions:
        propositions.append(Document(page_content=proposition, metadata={"Title": "Paul Graham's Founder Mode Essay", "Source": "https://www.perplexity.ai/page/paul-graham-s-founder-mode-ess-t9TCyvkqRiyMQJWsHr0fnQ", "chunk_id": i+1}))

"""
### Quality Check
"""
logger.info("### Quality Check")

class GradePropositions(BaseModel):
    """Grade a given proposition on accuracy, clarity, completeness, and conciseness"""

    accuracy: int = Field(
        description="Rate from 1-10 based on how well the proposition reflects the original text."
    )

    clarity: int = Field(
        description="Rate from 1-10 based on how easy it is to understand the proposition without additional context."
    )

    completeness: int = Field(
        description="Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers)."
    )

    conciseness: int = Field(
        description="Rate from 1-10 based on whether the proposition is concise without losing important information."
    )

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
structured_llm= llm.with_structured_output(GradePropositions)

evaluation_prompt_template = """
Please evaluate the following proposition based on the criteria below:
- **Accuracy**: Rate from 1-10 based on how well the proposition reflects the original text.
- **Clarity**: Rate from 1-10 based on how easy it is to understand the proposition without additional context.
- **Completeness**: Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers).
- **Conciseness**: Rate from 1-10 based on whether the proposition is concise without losing important information.

Example:
Docs: In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.

Propositons_1: Neil Armstrong was an astronaut.
Evaluation_1: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_2: Neil Armstrong walked on the Moon in 1969.
Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_3: Neil Armstrong was the first person to walk on the Moon.
Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_4: Neil Armstrong walked on the Moon during the Apollo 11 mission.
Evaluation_4: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_5: The Apollo 11 mission occurred in 1969.
Evaluation_5: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Format:
Proposition: "{proposition}"
Original Text: "{original_text}"
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", evaluation_prompt_template),
        ("human", "{proposition}, {original_text}"),
    ]
)

proposition_evaluator = prompt | structured_llm

evaluation_categories = ["accuracy", "clarity", "completeness", "conciseness"]
thresholds = {"accuracy": 7, "clarity": 7, "completeness": 7, "conciseness": 7}

def evaluate_proposition(proposition, original_text):
    response = proposition_evaluator.invoke({"proposition": proposition, "original_text": original_text})

    scores = {"accuracy": response.accuracy, "clarity": response.clarity, "completeness": response.completeness, "conciseness": response.conciseness}  # Implement function to extract scores from the LLM response
    return scores

def passes_quality_check(scores):
    for category, score in scores.items():
        if score < thresholds[category]:
            return False
    return True

evaluated_propositions = [] # Store all the propositions from the document

for idx, proposition in enumerate(propositions):
    scores = evaluate_proposition(proposition.page_content, doc_splits[proposition.metadata['chunk_id'] - 1].page_content)
    if passes_quality_check(scores):
        evaluated_propositions.append(proposition)
    else:
        logger.debug(f"{idx+1}) Propostion: {proposition.page_content} \n Scores: {scores}")
        logger.debug("Fail")

"""
### Embedding propositions in a vectorstore
"""
logger.info("### Embedding propositions in a vectorstore")

vectorstore_propositions = FAISS.from_documents(evaluated_propositions, embedding_model)
retriever_propositions = vectorstore_propositions.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}, # number of documents to retrieve
            )

query = "Who's management approach served as inspiartion for Brian Chesky's \"Founder Mode\" at Airbnb?"
res_proposition = retriever_propositions.invoke(query)

for i, r in enumerate(res_proposition):
    logger.debug(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

"""
### Comparing performance with larger chunks size
"""
logger.info("### Comparing performance with larger chunks size")

vectorstore_larger = FAISS.from_documents(doc_splits, embedding_model)
retriever_larger = vectorstore_larger.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}, # number of documents to retrieve
            )

res_larger = retriever_larger.invoke(query)

for i, r in enumerate(res_larger):
    logger.debug(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

"""
### Testing

#### Test - 1
"""
logger.info("### Testing")

test_query_1 = "what is the essay \"Founder Mode\" about?"
res_proposition = retriever_propositions.invoke(test_query_1)
res_larger = retriever_larger.invoke(test_query_1)

for i, r in enumerate(res_proposition):
    logger.debug(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

for i, r in enumerate(res_larger):
    logger.debug(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

"""
#### Test - 2
"""
logger.info("#### Test - 2")

test_query_2 = "who is the co-founder of Airbnb?"
res_proposition = retriever_propositions.invoke(test_query_2)
res_larger = retriever_larger.invoke(test_query_2)

for i, r in enumerate(res_proposition):
    logger.debug(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

for i, r in enumerate(res_larger):
    logger.debug(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

"""
#### Test - 3
"""
logger.info("#### Test - 3")

test_query_3 = "when was the essay \"founder mode\" published?"
res_proposition = retriever_propositions.invoke(test_query_3)
res_larger = retriever_larger.invoke(test_query_3)

for i, r in enumerate(res_proposition):
    logger.debug(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

for i, r in enumerate(res_larger):
    logger.debug(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

"""
### Comparison

| **Aspect**                | **Proposition-Based Retrieval**                                         | **Simple Chunk Retrieval**                                              |
|---------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **Precision in Response**  | High: Delivers focused and direct answers.                              | Medium: Provides more context but may include irrelevant information.    |
| **Clarity and Brevity**    | High: Clear and concise, avoids unnecessary details.                    | Medium: More comprehensive but can be overwhelming.                      |
| **Contextual Richness**    | Low: May lack context, focusing on specific propositions.               | High: Provides additional context and details.                           |
| **Comprehensiveness**      | Low: May omit broader context or supplementary details.                 | High: Offers a more complete view with extensive information.            |
| **Narrative Flow**         | Medium: Can be fragmented or disjointed.                                | High: Preserves the logical flow and coherence of the original document. |
| **Information Overload**   | Low: Less likely to overwhelm with excess information.                  | High: Risk of overwhelming the user with too much information.           |
| **Use Case Suitability**   | Best for quick, factual queries.                                        | Best for complex queries requiring in-depth understanding.               |
| **Efficiency**             | High: Provides quick, targeted responses.                               | Medium: May require more effort to sift through additional content.      |
| **Specificity**            | High: Precise and targeted responses.                                   | Medium: Answers may be less targeted due to inclusion of broader context.|
"""
logger.info("### Comparison")

logger.info("\n\n[DONE]", bright=True)