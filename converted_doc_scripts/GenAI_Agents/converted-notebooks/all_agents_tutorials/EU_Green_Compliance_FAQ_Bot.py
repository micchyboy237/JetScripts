from IPython.display import Image, display
from collections import Counter
from dotenv import load_dotenv
from jet.llm.mlx.adapters.mlx_langchain_llm_adapter import ChatMLX
from jet.llm.ollama.base_langchain.embeddings import MLXEmbeddings
from jet.logger import CustomLogger
from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatMLX
from langchain.llms import MLX
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from openai import MLX
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import base64
import json
import numpy as np
import openai
import os
import os  # Add this import first
import requests
import shutil
import sys
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/Avtr99/GenAI_Agents/blob/main/EU_Green_Compliance_FAQ_Bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **EU Green deal compliance FAQ Bot**

A RAG based AI agent that helps SMEs/ businesses quickly find answers to common questions about EU green deal policies. This bot will focus on responding to frequently asked questions (FAQs) related to the most relevant regulations, providing short and clear answers to help businesses understand and meet compliance standards.



**Functionality:** The bot answers basic questions about key EU environmental regulations, focusing on common requirements like waste management, carbon footprint reporting, and renewable energy.

# **Motivation**

Navigating EU green compliance can be overwhelming for businesses, especially smaller ones without dedicated resources. The project aims to simplify this process by creating a smart, accessible FAQ bot that provides instant, accurate answers to common questions about the EU Green Deal, emissions reporting, and waste management. By helping businesses understand and meet green regulations, compliance easier—it will contribute to a more sustainable future for everyone.

# **Method Details**

### **Document Storage and Embedding:**
Large documents are preprocessed into manageable chunks using a LLM for semantic chunking and stored in a vectorstore.
### **Query Processing:**
User queries are first rephrased to improve clarity and intent matching. The rephrased queries are then embedded using the same model. Using vector similarity and semantic relevance, the system retrieves the most relevant document chunks from the FAISS vectorstore.

### **Summarization:**
Context-aware and concise response are generated from the retrieved chunks using an LLM. This summarization step emphasizes clarity and ensures the answer directly aligns with the user’s query, distilling only the most relevant information.
### **Evaluation:**
Generated answers are evaluated against a gold Q&A dataset for factual accuracy and contextual relevance. The evaluation process includes metrics such as cosine similarity, F1 score, and semantic match.
### **Key Agents:**
Retriever Agent:
Retrieves the most semantically relevant chunks from the FAISS vectorstore based on the processed and embedded user query

Summarizer Agent:
Generate a coherent, concise response based on retrieved content.

Evaluation Agent:
Evaluates the quality of the generated response using gold-standard answers and similarity metrics.

# **Benefits of the Approach**

### **Accuracy and Fact-Checking:**
Reduces hallucination by grounding answers in external knowledge.

### **Modularity:**
The system's components (retriever, summarizer, evaluator) are independently - designed, allowing seamless improvements or replacements as needed.

### **Better evaluation:**
Combines advanced metrics like cosine similarity and F1 scores with gold q&a benchmark.

### **Flexibility:**
Adaptable across various domains and use cases with minimal pipeline changes, accommodating tailored retriever and summarizer configurations.

### **Context-Aware Responses:**
Incorporates context from both the query and the retrieved information.

# **Setup**

Import the required libraries
"""
logger.info("# **EU Green deal compliance FAQ Bot**")

# !pip install langchain langchain-openai python-dotenv openai
pip install langchain-experimental
pip install faiss-cpu


# os.environ["OPENAI_API_KEY"] = "ADD your key here" #set an openAI key

"""
initialize language model
"""
logger.info("initialize language model")

llm = ChatMLX(model="llama-3.2-3b-instruct", max_tokens=1000, temperature=0.7)

"""
# **Graph**
"""
logger.info("# **Graph**")


def render_mermaid(graph_definition: str, width: int = 800, height: int = 600):
    """
    Render a mermaid graph as an image using mermaid.ink and scale it.

    Args:
        graph_definition (str): The mermaid graph definition in string format.
        width (int): Desired width of the graph.
        height (int): Desired height of the graph.
    """
    graph_bytes = graph_definition.encode("utf-8")
    base64_bytes = base64.urlsafe_b64encode(graph_bytes)
    base64_string = base64_bytes.decode("ascii")
    image_url = f"https://mermaid.ink/img/{base64_string}"
    display(Image(url=image_url, width=width, height=height))


mermaid_graph = """
graph TD
    subgraph User_Query
        U[User Input Query] -->|Initiates Process| E[Rephrased Query]
    end
    subgraph Knowledge_Base_Processing
        A[EU Compliance Documents] -->|Text Splitter| B[Document Chunks]
        B -->|MLX Embedding| C[Vector Embeddings]
        C -->|Embeddings to Retriever| F[Retriever Agent]
    end
    subgraph Retriever_Agent
        E -->|Query Rephrasing| F[Processed Query]
        F -->|Vector Similarity Search| H[Retriever Search]
        H -->|Top-K Relevant Chunks| J[Retrieved Chunks]
    end
    subgraph Summarizer_Agent
        J -->|Contextual Summary| K[Context-Aware Summary]
        K -->|MLX LLM| L[Generated Summary]
        L -->|Summary for User| M[Final Summary]
    end
    subgraph Evaluation_Agent
        L -->|Evaluate Answer| N{Evaluation Metrics}
        P[(Gold Q&A Dictionary)] -->|Benchmark for Evaluation| N
        N -->|Cosine Similarity, F1 Score| O{Score Evaluation}
        N -->|Precision@1, Semantic Match| O
        O -->|Displayed Answer| M
    end
    M -->|Final Answer| T[User]
"""
render_mermaid(mermaid_graph, width=1200, height=1600)

"""
# Chunking the documents and Vector store

Semantic chunker using LLM and storing in a vectorstore
"""
logger.info("# Chunking the documents and Vector store")


folder_path = "/content/data"  # Path to the folder containing documents


def load_documents(folder_path):
    """
    Load and combine content from all text documents in the specified folder.

    Args:
        folder_path (str): Path to the folder containing documents.

    Returns:
        str: Combined content of all documents.
    """
    combined_content = ""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Adjust extensions as needed
        if os.path.isfile(file_path) and filename.endswith((".txt", ".md", ".docx")):
            with open(file_path, 'r', encoding='utf-8') as file:
                combined_content += file.read() + "\n"
    return combined_content


content = load_documents(folder_path)
if not content:
    raise ValueError("No valid documents found in the folder.")

# Specify the desired embedding model
embedding_model = MLXEmbeddings(model="mxbai-embed-large")
text_splitter = SemanticChunker(
    embeddings=embedding_model,  # Use the custom embedding model here
    # Use percentile-based semantic shifts for splitting
    breakpoint_threshold_type='percentile',
    # Define the threshold value (90th percentile)
    breakpoint_threshold_amount=90
)

docs = text_splitter.create_documents(
    [content])  # Semantic chunks as documents
logger.debug(f"Generated {len(docs)} semantic chunks.")

vectorstore = FAISS.from_documents(docs, embedding_model)

chunks_query_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3})  # Retrieve top-3 relevant chunks

query = "What are the goals of the European Green Deal?"
retrieved_chunks = chunks_query_retriever.invoke(query)

logger.debug("Retrieved Chunks for the Query:")
for idx, chunk in enumerate(retrieved_chunks, start=1):
    logger.debug(f"Chunk {idx}: {chunk.page_content}")

"""
# **Define the different functions for the collaboration system**

Next, the retriever agent should retreive the relevant chunks. Using both vector similarity and LLM-based grading.

## **Retriever agent**
"""
logger.info("# **Define the different functions for the collaboration system**")


class RetrieverAgent:
    def __init__(self, vectorstore, model="llama-3.2-3b-instruct", temperature=0.0):
        """
        Initialize the Retriever Agent with a FAISS vectorstore and MLX model.

        Args:
            vectorstore: FAISS vectorstore containing document chunks and their embeddings
            model (str): MLX model to use for relevance scoring (default: llama-3.2-3b-instruct)
        """
        self.vectorstore = vectorstore
        self.model = model
        self.temperature = temperature
#         openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure the MLX API key is set from environment variable

        self.llm = MLX(model=self.model, temperature=self.temperature)

        self.system = """You are a grader assessing relevance of a retrieved document to a user question.
                         If the document contains keyword(s) or semantic meaning related to the user question,
                         grade it as relevant.
                         It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
                         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    def _get_relevance_score(self, query: str, chunk_text: str) -> str:
        """
        Use the LLM with function call to grade the relevance of the chunk.

        Args:
            query (str): User query
            chunk_text (str): Text content of the chunk

        Returns:
            str: 'yes' or 'no' indicating whether the chunk is relevant or not
        """
        prompt = f"""Query: {query}
                    Chunk: {chunk_text}
                    Grade the relevance of this chunk to the query. Respond only with 'yes' or 'no'."""

        try:
            # Assuming llm has a generate method
            response = self.llm.generate([prompt])
            grade = response['choices'][0]['text'].strip()
            return grade.lower()

        except Exception as e:
            logger.debug(f"Error in grading: {e}")
            return "no"  # Default to no if there's an error

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3, rerank: bool = True) -> List[Dict]:
        """
        Retrieve and optionally rerank the most relevant chunks using both vector similarity
        and LLM-based grading.

        Args:
            query (str): User query
            top_k (int): Number of top relevant chunks to return
            rerank (bool): Whether to rerank results using LLM grading

        Returns:
            list: List of dictionaries containing similarity scores and chunk text
        """
        retrieved_docs = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k * (2 if rerank else 1)  # Get more candidates if reranking
        )

        logger.debug("Retrieved Docs (Raw):", retrieved_docs)

        relevant_chunks = []

        for doc, vector_score in retrieved_docs:
            chunk_info = {
                # Vector similarity score
                "vector_similarity": float(vector_score),
                "chunk_text": doc.page_content,
                "metadata": doc.metadata
            }

            if rerank:
                relevance_grade = self._get_relevance_score(
                    query, doc.page_content)

                if relevance_grade == "yes":
                    chunk_info["relevance_grade"] = relevance_grade
                    chunk_info["combined_score"] = 1 - \
                        vector_score  # Adjust this as necessary
                    relevant_chunks.append(chunk_info)
            else:
                chunk_info["combined_score"] = 1 - \
                    vector_score  # Adjust this as necessary
                relevant_chunks.append(chunk_info)

        relevant_chunks.sort(key=lambda x: x["combined_score"], reverse=True)
        return relevant_chunks[:top_k]

    def batch_retrieve(self, queries: List[str], top_k: int = 3, rerank: bool = True) -> Dict[str, List[Dict]]:
        """
        Batch process multiple queries.

        Args:
            queries (List[str]): List of queries to process
            top_k (int): Number of top relevant chunks to return per query
            rerank (bool): Whether to rerank results using LLM grading

        Returns:
            Dict[str, List[Dict]]: Dictionary mapping queries to their relevant chunks
        """
        results = {}
        for query in queries:
            results[query] = self.retrieve_relevant_chunks(
                query, top_k, rerank)
        return results


def create_retriever_agent(vectorstore, model="llama-3.2-3b-instruct", temperature=0.0):
    """
    Factory function to create a RetrieverAgent instance.

    Args:
        vectorstore: FAISS vectorstore containing document chunks
        model (str): MLX model to use for scoring (default: llama-3.2-3b-instruct)

    Returns:
        RetrieverAgent: Initialized retriever agent
    """
    return RetrieverAgent(vectorstore, model, temperature)


"""
## **Summarizer Agent**

Context aware summarization using LLM
"""
logger.info("## **Summarizer Agent**")


class SummarizerAgent:
    def __init__(self, model="llama-3.2-3b-instruct"):  # Default model can be adjusted
        """
        Initialize the Summarizer Agent with MLX model.

        Args:
            model (str): MLX model to use for summarization (default: llama-3.2-3b-instruct)
        """
        self.model = model
#         openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure the MLX API key is set from environment variable

    def summarize_text(self, query: str, text: str) -> str:
        """
        Summarize the given text in the context of the query, focusing on concise and clear details within two sentences.

        Args:
            query (str): User query.
            text (str): Text content to summarize.

        Returns:
            str: Concise and clear summary relevant to the query.
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            #             "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"  # Ensure the MLX API key is set
        }

        prompt = f"""Summarize the following text based on the query. Focus on extracting the most relevant details in a clear and concise manner, ensuring the summary is no more than two sentences.

        Query: {query}

        Text to summarize: {text}

        Please make sure the summary is brief, clear, and focuses on the key information, avoiding unnecessary details and providing a direct answer to the query.
        """

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a summarization assistant. Your task is to summarize text into two sentences, focusing on the key points and ensuring clarity and conciseness."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Low temperature for more focused responses
            "max_tokens": 150  # Ensure a concise summary
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an exception if the request fails

            result = response.json()
            summarized_text = result['choices'][0]['message']['content'].strip(
            )
            return summarized_text

        except requests.exceptions.RequestException as e:
            logger.debug(f"Error in summarization: {e}")
            return "Sorry, I could not generate the summary at the moment."

    def batch_summarize(self, queries: List[str], texts: List[str]) -> Dict[str, str]:
        """
        Batch process multiple queries and summarize corresponding texts.

        Args:
            queries (List[str]): List of queries to process.
            texts (List[str]): List of texts to summarize.

        Returns:
            Dict[str, str]: Dictionary mapping each query to its summarized text.
        """
        summaries = {}
        for query, text in zip(queries, texts):
            summaries[query] = self.summarize_text(query, text)
        return summaries


# Use the same model or another available model
summarizer = SummarizerAgent(model="llama-3.2-3b-instruct")

query = "What is the European Green Deal?"
text = """The European Green Deal is a set of policy initiatives by the European Commission to address climate change, promote sustainability, and reduce carbon emissions by 2030. The Deal includes measures to promote clean energy, sustainable agriculture, and investments in green technologies. It aims to make Europe the first carbon-neutral continent by 2050."""

summary = summarizer.summarize_text(query, text)
logger.debug(f"Summary: {summary}")

"""
# **Evaluation Agent**

Gold Q&A: List of curated question and answers that will be used to evaluted the answer
"""
logger.info("# **Evaluation Agent**")

gold_qa_dict = [
    {"query": "What is the European Green Deal (EGD)?", "answer": "The EGD is the EU’s strategy to reach net zero greenhouse gas emissions by 2050 while achieving sustainable economic growth. It covers policies across sectors like agriculture, energy, and manufacturing to ensure products meet higher sustainability standards."},
    {"query": "What is the Farm to Fork (F2F) Strategy?", "answer": "The F2F strategy is part of the EGD, focusing on making the EU’s food system fair, healthy, and environmentally friendly. It targets reducing pesticide use, nutrient loss, and promoting organic farming."},
    {"query": "What is the Circular Economy Action Plan (CEAP)?",
     "answer": "CEAP aims to eliminate waste by promoting the reuse, repair, and recycling of materials. It emphasizes creating sustainable products and reducing waste generation in industries like packaging, textiles, and electronics."},
    {"query": "What is the EU Green Deal Industrial Plan?", "answer": "The Plan aims to enhance Europe’s net-zero industrial base by simplifying regulations, increasing funding, developing skills, and fostering trade. It focuses on manufacturing key technologies like batteries, hydrogen systems, and wind turbines to achieve climate neutrality by 2050."},
    {"query": "What is the Net-Zero Industry Act (NZIA)?", "answer": "The NZIA aims to boost the EU's manufacturing capacity for net-zero technologies, such as solar panels, batteries, and electrolysers. It sets goals like manufacturing at least 40% of strategic net-zero technologies domestically by 2030."},
    {"query": "What is the EU Biodiversity Strategy for 2030?",
        "answer": "A key part of the Green Deal, it focuses on reversing biodiversity loss by restoring degraded ecosystems, reducing pesticide use by 50%, and ensuring 25% of farmland is organic by 2030."},
    {"query": "What is the Carbon Border Adjustment Mechanism (CBAM)?", "answer": "CBAM is a policy tool designed to prevent carbon leakage by imposing carbon costs on imports of certain goods from countries with less stringent climate policies. It ensures that imported products are priced similarly to EU-manufactured goods under the EU's carbon pricing system."},
    {"query": "Which sectors does CBAM initially cover?",
        "answer": "CBAM applies to high-emission sectors such as cement, iron and steel, fertilizers, electricity, and aluminum. Additional sectors may be included in the future."},
    {"query": "How does CBAM impact SMEs exporting to the EU?",
        "answer": "SMEs exporting CBAM-regulated goods must report the carbon emissions embedded in their products and potentially pay a carbon price. This may require investment in cleaner technologies and better transparency in production processes."},
    {"query": "When will CBAM come into effect?",
        "answer": "CBAM will be implemented in stages, starting with a reporting phase in 2023 and transitioning to full operation with financial obligations by 2026."},
    {"query": "How can exporters mitigate CBAM costs?",
        "answer": "Exporters can invest in low-carbon production methods or provide evidence of carbon taxes already paid in their home countries to reduce or eliminate CBAM charges."},
    {"query": "What sustainability standards must SMEs exporting to the EU meet?",
        "answer": "SMEs must meet standards for reduced waste, traceable production, eco-friendly packaging, and compliance with the new Ecodesign for Sustainable Products Regulation."},
    {"query": "What are the traceability requirements for exporters?",
        "answer": "Exporters must provide detailed information on product life cycles, including manufacturing, materials used, and compliance with sustainability criteria."},
    {"query": "How does the Carbon Border Adjustment Mechanism (CBAM) affect imports?",
     "answer": "CBAM imposes carbon taxes on imported goods with high greenhouse gas footprints, ensuring imports align with EU environmental standards."},
    {"query": "What is required under the new EU organic regulations?",
        "answer": "Imported organic products must display control body codes, follow strict organic certification rules, and meet labeling requirements."},
    {"query": "How does the Green Deal Industrial Plan simplify regulations for SMEs?",
        "answer": "The Plan introduces streamlined permitting processes and 'one-stop shops' to reduce red tape for projects related to renewable technologies."},
    {"query": "What is the Digital Product Passport (DPP)?",
     "answer": "The DPP provides detailed information about a product’s lifecycle, ensuring traceability and compliance with sustainability standards. It helps SMEs align with EU buyers' expectations."},
    {"query": "What are the biodiversity-related commitments for agricultural land?",
        "answer": "By 2030, 10% of farmland must feature biodiversity-friendly measures, and pesticide use must be cut by 50%."},
    {"query": "What challenges might SMEs face due to the EGD?",
        "answer": "SMEs may encounter higher production costs, complex sustainability reporting requirements, and the need to adapt to new eco-friendly technologies."},
    {"query": "What are the compliance deadlines for key regulations?",
        "answer": "Major regulations like the revision of pesticide use directives and the CBAM will be implemented in stages, with some taking effect by 2024."},
    {"query": "How does the EU support skill development for the green transition?",
        "answer": "The EU is establishing Net-Zero Industry Academies to train workers in net-zero technologies, with funding for reskilling and upskilling programs."},
    {"query": "What is the timeline for major Green Deal initiatives?",
        "answer": "Key initiatives like the NZIA and biodiversity commitments have milestones up to 2030, with significant mid-term reviews and funding disbursements expected between 2023 and 2026."},
    {"query": "What funding mechanisms are available for SMEs under the Green Deal?",
        "answer": "SMEs can access funding through programs like the Innovation Fund, InvestEU, and the European Sovereignty Fund. These mechanisms support green technology projects and offer tax breaks."},
    {"query": "What is the European Hydrogen Bank?",
        "answer": "It is a financial instrument to support renewable hydrogen production and imports. The Bank offers subsidies to bridge the cost gap between renewable and fossil hydrogen."},
    {"query": "What trade opportunities does the Green Deal provide?",
        "answer": "The Plan promotes open and fair trade through partnerships, free trade agreements, and initiatives like the Critical Raw Materials Club to ensure supply chain resilience."},
    {"query": "How can SMEs benefit from the EU Green Deal?",
        "answer": "SMEs can capitalize on increased demand for sustainable products, gain partnerships with EU companies, and access new markets driven by sustainability goals."},
    {"query": "What support is available for SMEs transitioning to sustainable practices?",
        "answer": "EU-based programs provide subsidies, technical support, and resources like the Digital Product Passport to help SMEs adapt."},
    {"query": "What opportunities do CEAP and F2F provide?",
        "answer": "These initiatives create markets for sustainable products, such as organic food and recycled textiles, enhancing SME competitiveness."},
    {"query": "What is the role of the EU Digital Product Passport?",
        "answer": "This tool standardizes and simplifies compliance, providing detailed product information to buyers while promoting transparency."},
    {"query": "What are Net-Zero Strategic Projects?",
        "answer": "These are priority projects essential for the EU's energy transition, such as large-scale solar or battery manufacturing plants. They benefit from accelerated permitting and funding."},
    {"query": "How does the EU address biodiversity in urban planning?",
        "answer": "Through the Green City Accord, urban planning integrates green spaces and biodiversity-focused infrastructure."},
    {"query": "What role does hydrogen play in the EU's climate strategy?",
        "answer": "Hydrogen is a cornerstone for reducing industrial emissions, with a target of producing 10 million tonnes of renewable hydrogen in the EU and importing an additional 10 million tonnes by 2030."},
    {"query": "What are the packaging requirements under the EGD?",
        "answer": "All packaging must be reusable or recyclable by 2024, with reduced material complexity and increased recycled content."},
    {"query": "How does the EU Biodiversity Strategy impact exporters?",
        "answer": "Exporters must ensure their products do not contribute to deforestation or biodiversity loss and comply with due diligence laws."}
]

"""
**Evaluation Agent:** Evaluates the generated answer
"""


class EvaluationAgent:
    def __init__(self, gold_qa_dict, similarity_threshold=0.85):
        """
        Initialize the Evaluation Agent with a cosine similarity-based approach.

        Args:
            gold_qa_dict (list): A list of dictionaries containing gold Q&A where each
                                  dictionary has keys "query" and "answer".
            similarity_threshold (float): Minimum cosine similarity score to accept an answer
                                           without human review (default is 0.85).
        """
        self.gold_qa_dict = gold_qa_dict
        self.similarity_threshold = similarity_threshold

    def _tokenize_text(self, text):
        """
        Tokenize the text by splitting it into words and converting to lowercase.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of tokens (words).
        """
        return text.lower().split()

    def _vectorize_text(self, text):
        """
        Convert tokenized text into a term frequency (TF) vector.

        Args:
            text (str): The text to vectorize.

        Returns:
            dict: Term frequency (TF) vector.
        """
        tokens = self._tokenize_text(text)
        return Counter(tokens)

    def _cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two term frequency vectors.

        Args:
            vec1 (dict): Term frequency vector of the first text.
            vec2 (dict): Term frequency vector of the second text.

        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        all_tokens = set(vec1.keys()).union(set(vec2.keys()))
        vec1_list = [vec1.get(token, 0) for token in all_tokens]
        vec2_list = [vec2.get(token, 0) for token in all_tokens]

        return cosine_similarity([vec1_list], [vec2_list])[0][0]

    def _calculate_f1_score(self, generated_answer, gold_answer):
        """
        Calculate F1 score based on token overlap between generated and gold answers.

        Args:
            generated_answer (str): The answer generated by the system.
            gold_answer (str): The gold standard answer.

        Returns:
            float: F1 score based on token overlap.
        """
        gen_tokens = set(self._tokenize_text(generated_answer))
        gold_tokens = set(self._tokenize_text(gold_answer))

        precision = len(gen_tokens & gold_tokens) / \
            len(gen_tokens) if len(gen_tokens) > 0 else 0
        recall = len(gen_tokens & gold_tokens) / \
            len(gold_tokens) if len(gold_tokens) > 0 else 0

        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0
        return f1

    def evaluate_answer(self, generated_answer, query):
        """
        Evaluate the generated answer using multiple metrics including F1 score, Precision@1, and cosine similarity.

        Args:
            generated_answer (str): The answer generated by the system.
            query (str): The user query to evaluate.

        Returns:
            dict: Evaluation results with various metrics.
        """
        normalized_query = query.strip().lower()

        gold_answer = None
        for qa in self.gold_qa_dict:
            gold_query = qa["query"].strip().lower()
            if normalized_query == gold_query:
                gold_answer = qa["answer"]
                break

        if not gold_answer:
            return {"error": "No Gold Standard: The query is not in the gold Q&A dictionary."}

        gen_vec = self._vectorize_text(generated_answer)
        gold_vec = self._vectorize_text(gold_answer)

        cosine_sim = self._cosine_similarity(gen_vec, gold_vec)

        f1 = self._calculate_f1_score(generated_answer, gold_answer)

        semantic_match = cosine_sim >= self.similarity_threshold
        precision_at_1 = 1 if semantic_match else 0

        human_review_needed = cosine_sim < self.similarity_threshold

        return {
            "cosine_similarity": cosine_sim,
            "f1_score": f1,
            "precision_at_1": precision_at_1,
            "semantic_match": semantic_match,
            "human_review_needed": human_review_needed,
            "generated_answer": generated_answer,
            "gold_answer": gold_answer
        }


gold_qa_dict = [
    {"query": "What is the European Green Deal (EGD)?", "answer":
     "The EGD is the EU’s strategy to reach net zero greenhouse gas emissions by 2050 while achieving sustainable economic growth. It covers policies across sectors like agriculture, energy, and manufacturing to ensure products meet higher sustainability standards."},
    {"query": "What is the Farm to Fork strategy (F2F)?", "answer":
     "The F2F strategy is part of the European Green Deal, focusing on making the EU’s food system fair, healthy, and environmentally friendly. It targets reducing pesticide use, nutrient loss, and promoting organic farming."}
]

evaluation_agent = EvaluationAgent(gold_qa_dict, similarity_threshold=0.85)

generated_answer = "The F2F strategy is part of the EGD, focusing on making the EU’s food system fair, healthy, and environmentally friendly. It targets reducing pesticide use, nutrient loss, and promoting organic farming."
user_question = "What is the Farm to Fork strategy (F2F)?"

evaluation_result = evaluation_agent.evaluate_answer(
    generated_answer, user_question)

logger.debug(
    f"Cosine Similarity: {evaluation_result['cosine_similarity']:.2f}")
logger.debug(f"F1 Score (Overlap): {evaluation_result['f1_score']:.2f}")
logger.debug(f"Precision@1: {evaluation_result['precision_at_1']}")
logger.debug(f"Semantic Match: {evaluation_result['semantic_match']}")
logger.debug(
    f"Human Review Needed: {evaluation_result['human_review_needed']}")
logger.debug(f"Generated Answer: {evaluation_result['generated_answer']}")
logger.debug(f"Gold Answer: {evaluation_result['gold_answer']}")

"""
# **RelevanceSummarySystem Class**

Brings together all the agents. There is also a rephraser function, that rephrases the user query for better retrieval accuracy
"""
logger.info("# **RelevanceSummarySystem Class**")


class RelevanceSummarizationSystem:
    def __init__(self, retriever_agent, summarizer_agent, evaluation_agent, relevance_threshold=0.6, openai_api_key=None):
        """
        Initialize the Relevance Summarization System.
        """
        self.retriever_agent = retriever_agent
        self.summarizer_agent = summarizer_agent
        self.evaluation_agent = evaluation_agent
        self.relevance_threshold = relevance_threshold
#         self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("MLX API key is required for rephrasing queries.")

    def _send_openai_request(self, prompt: str, model="llama-3.2-3b-instruct", temperature=0.7, max_tokens=150):
        """
        Helper function to send a request to MLX's API and handle the response.
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            logger.debug(f"❌ Error during API request: {e}")
            return None

    def rephrase_query(self, query: str) -> str:
        """
        Rephrase the query using MLX's API to improve retrieval accuracy.
        """
        prompt = f"You are a rephrasing expert. Rephrase the following question to make it clearer and more likely to retrieve relevant information: {query}"
        rephrased_query = self._send_openai_request(
            prompt, model="llama-3.2-3b-instruct", max_tokens=60)

        if rephrased_query:
            logger.debug(f"🔄 Rephrased query: {rephrased_query}")
            return rephrased_query
        return query  # Fallback to the original query if rephrasing fails

    def process_query(self, query: str, top_k: int = 3):
        """
        Process a user query by retrieving relevant chunks and summarizing them.
        """
        logger.debug(f"🔍 Processing query: {query}\n")

        rephrased_query = self.rephrase_query(query)

        try:
            original_chunks = self.retriever_agent.retrieve_relevant_chunks(
                query, top_k=top_k)
            rephrased_chunks = self.retriever_agent.retrieve_relevant_chunks(
                rephrased_query, top_k=top_k)
        except Exception as e:
            logger.debug(f"❌ Error during retrieval: {e}")
            return "An error occurred while processing your query. Please try again later."

        all_chunks = sorted(original_chunks + rephrased_chunks,
                            key=lambda x: x["combined_score"], reverse=True)

        if not all_chunks:
            logger.debug("⚠️ No relevant chunks found.\n")
            return "I don't know the answer to this question. Can you try rephrasing your question and try again?"

        top_relevance = all_chunks[0]["combined_score"]
        logger.debug(f"📊 Top relevance score: {top_relevance:.2f}")

        if top_relevance < self.relevance_threshold:
            logger.debug(
                f"⚠️ Relevance score too low (Score: {top_relevance:.2f}).\n")
            return "I don't know the answer to this question. Can you try rephrasing your question and try again?"

        try:
            summary = self.summarizer_agent.summarize_retrieved_chunks(
                all_chunks, query)
        except Exception as e:
            logger.debug(f"❌ Error during summarization: {e}")
            return "An error occurred while summarizing the information. Please try again later."

        evaluation_result = self.evaluation_agent.evaluate_answer(
            summary, query)

        logger.debug(f"📝 Evaluation Results: {evaluation_result}\n")

        return summary.strip(), evaluation_result


"""
# **Example Usage**

Try executing the code to type your question
"""
logger.info("# **Example Usage**")

evaluation_agent = EvaluationAgent(gold_qa_dict)

relevance_system = RelevanceSummarizationSystem(
    retriever_agent=retriever_agent,  # Assuming this is already defined
    summarizer_agent=summarizer_agent,  # Assuming this is already defined
    evaluation_agent=evaluation_agent,
    relevance_threshold=0.6
)

user_question = input("Enter your question: ")  # User-provided query

final_summary, evaluation_results = relevance_system.process_query(
    user_question, top_k=3)

logger.debug("\nResponse:")
logger.debug(final_summary)  # Clean and concise summary
logger.debug("\nEvaluation Results:")
logger.debug(evaluation_results)  # Evaluation metrics

logger.info("\n\n[DONE]", bright=True)
