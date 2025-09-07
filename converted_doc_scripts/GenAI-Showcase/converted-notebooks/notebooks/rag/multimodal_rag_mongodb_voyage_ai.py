from PIL import Image
from google import genai
from google.cloud import storage
from google.genai import types
from io import BytesIO
from jet.logger import CustomLogger
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_random_exponential
from tqdm import tqdm
from typing import List
from voyageai import Client
import json
import os
import pymupdf
import re
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/multimodal_rag_mongodb_voyage_ai.ipynb)

# Building Multimodal RAG Applications with MongoDB and Voyage AI

In this notebook, you will learn how to build multimodal RAG applications using Voyage AI's multimodal embedding models and Google's multimodal LLMs.

Additionally, we will evaluate Voyage AI's VLM-based embedding model against CLIP-based embedding models on our dataset.

# Step 1: Install required libraries

* **pymongo**: Python driver for MongoDB
* **voyageai**: Python client for Voyage AI
* **google-genai**: Python library to access Google's embedding models and LLMs via Google AI Studio
* **google-cloud-storage**: Python client for Google Cloud Storage
* **sentence-transformers**: Python library to use open-source ML models from Hugging Face
* **PyMuPDF**: Python library for analyzing and manipulating PDFs
* **Pillow**: A Python imaging library
* **tqdm**: Show progress bars for loops in Python
* **tenacity**: Python library for easily adding retries to functions
"""
logger.info("# Building Multimodal RAG Applications with MongoDB and Voyage AI")

# !pip install -qU pymongo voyageai google-genai google-cloud-storage sentence-transformers PyMuPDF Pillow tqdm tenacity

"""
# Step 2: Setup prerequisites

* **Set the MongoDB connection string**: Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.
* **Set the Voyage AI API key**: Follow the steps [here](https://docs.voyageai.com/docs/api-key-and-installation#authentication-with-api-keys) to get a Voyage AI API key.
* **Set a Gemini API key**: Follow the steps [here](https://ai.google.dev/gemini-api/docs/api-key) to get a Gemini API key via Google AI Studio.
* [In a separate terminal] **Setup Application Default Credentials (ADC)**: Follow the steps [here](https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment#google-idp) to configure ADC via the Google Cloud CLI.
"""
logger.info("# Step 2: Setup prerequisites")

# import getpass

"""
### MongoDB
"""
logger.info("### MongoDB")

# MONGODB_URI = getpass.getpass("Enter your MongoDB connection string: ")

"""
### Voyage AI
"""
logger.info("### Voyage AI")

# os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter your Voyage AI API key: ")

"""
### Google
"""
logger.info("### Google")

# GEMINI_API_KEY = getpass.getpass("Enter your Gemini API key: ")

"""
# Step 3: Read PDF from URL
"""
logger.info("# Step 3: Read PDF from URL")



response = requests.get("https://arxiv.org/pdf/2501.12948")
if response.status_code != 200:
    raise ValueError(f"Failed to download PDF. Status code: {response.status_code}")
pdf_stream = BytesIO(response.content)
pdf = pymupdf.open(stream=pdf_stream, filetype="pdf")

"""
# Step 4: Store PDF images in GCS and extract metadata for MongoDB
"""
logger.info("# Step 4: Store PDF images in GCS and extract metadata for MongoDB")


GCS_PROJECT = "genai"
GCS_BUCKET = "tutorials"

gcs_client = storage.Client(project=GCS_PROJECT)
gcs_bucket = gcs_client.bucket(GCS_BUCKET)

def upload_image_to_gcs(key: str, data: bytes) -> None:
    """
    Upload image to GCS.

    Args:
        key (str): Unique identifier for the image in the bucket.
        data (bytes): Image bytes to upload.
    """
    blob = gcs_bucket.blob(key)
    blob.upload_from_string(data, content_type="image/png")

docs = []

zoom = 3.0
mat = pymupdf.Matrix(zoom, zoom)
for n in tqdm(range(pdf.page_count)):
    temp = {}
    pix = pdf[n].get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    gcs_key = f"multimodal-rag/{n+1}.png"
    upload_image_to_gcs(gcs_key, img_bytes)
    temp["image"] = img_bytes
    temp["gcs_key"] = gcs_key
    temp["width"] = pix.width
    temp["height"] = pix.height
    docs.append(temp)

"""
# Step 5: Add embeddings to the MongoDB documents
"""
logger.info("# Step 5: Add embeddings to the MongoDB documents")



voyageai_client = Client()

clip_model = SentenceTransformer("clip-ViT-B-32")

def get_voyage_embedding(data: Image.Image | str, input_type: str) -> List:
    """
    Get Voyage AI embeddings for images and text.

    Args:
        data (Image.Image | str): An image or text to embed.
        input_type (str): Input type, either "document" or "query".

    Returns:
        List: Embeddings as a list.
    """
    embedding = voyageai_client.multimodal_embed(
        inputs=[[data]], model="voyage-multimodal-3", input_type=input_type
    ).embeddings[0]
    return embedding

def get_clip_embedding(data: Image.Image | str) -> List:
    """
    Get CLIP embeddings for images and text.

    Args:
        data (Image.Image | str): An image or text to embed.

    Returns:
        List: Embeddings as a list.
    """
    embedding = clip_model.encode(data).tolist()
    return embedding

embedded_docs = []

for doc in tqdm(docs):
    img = Image.open(BytesIO(doc["image"]))
    doc["voyage_embedding"] = get_voyage_embedding(img, "document")
    doc["clip_embedding"] = get_clip_embedding(img)
    del doc["image"]
    embedded_docs.append(doc)

embedded_docs[0].keys()

"""
# Step 6: Ingest documents into MongoDB
"""
logger.info("# Step 6: Ingest documents into MongoDB")


mongodb_client = MongoClient(
    MONGODB_URI, appname="devrel.showcase.multimodal_rag_mongodb_voyage_ai"
)
mongodb_client.admin.command("ping")

DB_NAME = "mongodb"
COLLECTION_NAME = "multimodal_rag"
VS_INDEX_NAME = "vector_index"

collection = mongodb_client[DB_NAME][COLLECTION_NAME]

collection.delete_many({})

collection.insert_many(embedded_docs)

"""
# Step 7: Create a vector search index
"""
logger.info("# Step 7: Create a vector search index")

model = {
    "name": VS_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "voyage_embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            },
            {
                "type": "vector",
                "path": "clip_embedding",
                "numDimensions": 512,
                "similarity": "cosine",
            },
        ]
    },
}

collection.create_search_index(model=model)

"""
# Step 8: Retrieve documents using vector search
"""
logger.info("# Step 8: Retrieve documents using vector search")

def get_image_from_gcs(key: str) -> bytes:
    """
    Get image bytes from GCS.

    Args:
        key (str): Identifier for the image in the bucket.

    Returns:
        bytes: Image bytes.
    """
    blob = gcs_bucket.blob(key)
    image_bytes = blob.download_as_bytes()
    return image_bytes

def vector_search(
    user_query: str, model: str, display_images: bool = True
) -> List[str]:
    """
    Perform vector search and display images, and return the GCS keys.

    Args:
        user_query (str): User query string.
        model (str): Embedding model to use, either "voyage" or "clip".
        display_images (bool, optional): Whether or not to display images. Defaults to True.

    Returns:
        List[str]: List of GCS keys.
    """
    if model == "voyage":
        query_embedding = get_voyage_embedding(user_query, "query")
    elif model == "clip":
        query_embedding = get_clip_embedding(user_query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": VS_INDEX_NAME,
                "queryVector": query_embedding,
                "path": f"{model}_embedding",
                "numCandidates": 150,
                "limit": 5,
            }
        },
        {
            "$project": {
                "_id": 0,
                "gcs_key": 1,
                "width": 1,
                "height": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = collection.aggregate(pipeline)

    gcs_keys = []
    for result in results:
        gcs_key = result["gcs_key"]
        if display_images is True:
            img = Image.open(BytesIO(get_image_from_gcs(gcs_key)))
            logger.debug(f"{result['score']}\n")
            display(img)
        gcs_keys.append(gcs_key)

    return gcs_keys

vector_search(
    "Summarize the Pass@1 accuracy of Deepseek R1 against other models.",
    "voyage",
    display_images=True,
)

vector_search(
    "Summarize the Pass@1 accuracy of Deepseek R1 against other models.",
    "clip",
    display_images=True,
)

"""
# Step 9: Create a multimodal RAG app
"""
logger.info("# Step 9: Create a multimodal RAG app")


gemini_client = genai.Client(api_key=GEMINI_API_KEY)

LLM = "gemini-2.0-flash"

def generate_answer(user_query: str, model: str) -> str:
    """
    Generate answer to the user question using a Gemini multimodal LLM.

    Args:
        user_query (str): User query string.
        model (str): Embedding model to use, either "voyage" or "clip".

    Returns:
        str: LLM generated answer.
    """
    gcs_keys = vector_search(user_query, model, display_images=False)
    images = [Image.open(BytesIO(get_image_from_gcs(key))) for key in gcs_keys]
    prompt = f"Answer the question based only on the provided context. If the context is empty, say I DON'T KNOW\n\nQuestion:{user_query}\n\nContext:\n"
    messages = [prompt] + images
    response = gemini_client.models.generate_content(
        model=LLM,
        contents=messages,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return response.text

generate_answer(
    "Summarize the Pass@1 accuracy of Deepseek R1 against other models.", "voyage"
)

generate_answer(
    "Summarize the Pass@1 accuracy of Deepseek R1 against other models.", "clip"
)

"""
# Step 10: Evaluating retrieval and generation
"""
logger.info("# Step 10: Evaluating retrieval and generation")



eval_dataset = [
    {
        "question": "How does DeepSeek-R1-Zero's accuracy on the AIME 2024 benchmark improve during RL training?",
        "answer": "The accuracy measured by the average pass@1 score increases from 15.6% to 71.0% pass@1 as a result of RL training.",
        "page_numbers": [6, 7, 3, 1, 13],
    },
    {
        "question": "How does DeepSeek-R1-Zero compare to Ollama models on the MATH-500 benchmark?",
        "answer": "DeepSeek-R1-Zero achieves 95.9% pass@1, outperforming Ollama-o1-0912 at 94.8% as well as Ollama-o1-mini at 90%.",
        "page_numbers": [7, 1, 4, 13, 14],
    },
    {
        "question": "What trend is observed in the model's response length during RL training?",
        "answer": "Response length increases consistently as RL training progresses owing to more reasoning tokens used as the model learns to explore and refine its thought processes in greater depth.",
        "page_numbers": [8, 7, 6, 3, 9],
    },
    {
        "question": "How does DeepSeek-R1 perform on coding tasks compared to Ollama-o1-1217?",
        "answer": "DeepSeek-R1 scores higher LiveCodeBench and SWE Verified but lower on Codeforces and Aider-Polyglot.",
        "page_numbers": [13, 14, 1, 3, 7],
    },
    {
        "question": "Summarize the performance of distilled 32B models on the AIME 2024 benchmark?",
        "answer": "DeepSeek-R1-Distill-Qwen 14B and 32B, and DeepSeek-R1-Distill-Llama 8B and 70B surpass models such as GPT-4o-0513, Claude-3.5-Sonnet-1022, Ollama-o1-mini and QwQ-32B-Preview on the AIME 2024 benchmark.",
        "page_numbers": [15, 14, 4, 3, 11],
    },
    {
        "question": "What reinforcement learning algorithm replaces traditional critic models in the paper?",
        "answer": "Group Relative Policy Optimization (GRPO), which uses group scores to estimate baselines instead.",
        "page_numbers": [5, 6, 3, 4, 15],
    },
    {
        "question": "How does DeepSeek-R1 address readability issues present in DeepSeek-R1-Zero?",
        "answer": "It uses cold-start data and multi-stage training (SFT + RL) to improve output clarity.",
        "page_numbers": [9, 10, 5, 3, 4],
    },
    {
        "question": "What reward types guide DeepSeek-R1-Zero's RL training process?",
        "answer": "Accuracy rewards (correct answers) and format rewards (structured output with tags).",
        "page_numbers": [6, 10, 5, 15, 3],
    },
    {
        "question": "Why is Monte Carlo Tree Search (MCTS) challenging for reasoning tasks?",
        "answer": "Token generation creates an exponentially large search space, and value models are hard to train reliably.",
        "page_numbers": [15, 16, 5, 8, 11],
    },
    {
        "question": "What is supervised fine tuning?",
        "answer": "Supervised fine-tuning (SFT) refers to a training approach where a model is fine-tuned on specific data examples to improve its performance in targeted tasks.",
        "page_numbers": [10, 4, 9, 3, 5],
    },
]

def calculate_rr(ground_truth: List[int], retrieved: List[int]) -> float:
    """
    Calculate reciprocal rank for a given query.

    Args:
        ground_truth (List[int]): List of relevant page numbers.
        retrieved (List[int]): List of retrieved page numbers.

    Returns:
        float: Reciprocal rank.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in ground_truth:
            return 1 / (i + 1)
    return 0.0

def recall_at_k(ground_truth: List[int], retrieved: List[int], k: int = 1) -> float:
    """
    Calculate recall at k for a given query

    Args:
        ground_truth (List[int]): List of relevant page numbers.
        retrieved (List[int]): List of retrieved page numbers.
        k (int, optional): Position to calculate recall at. Defaults to 1.

    Returns:
        float: Recall at k.
    """
    retrieved_at_k = set(retrieved[:k])
    ground_truth_set = set(ground_truth)
    relevant_retrieved = len(retrieved_at_k.intersection(ground_truth_set))
    recall = relevant_retrieved / len(ground_truth_set)
    return recall

def get_page_num(gcs_key: str) -> int:
    """
    Extract page number from the GCS key.

    Args:
        gcs_key (str):GCS key for the image.

    Returns:
        int: Page number in the source PDF.
    """
    match = re.match(r"multimodal-rag/(\d+)\.png", gcs_key)
    return int(match.group(1))

@retry(wait=wait_random_exponential(multiplier=1, max=60))
def eval_alignment(query: str, ground_truth_answer: str, generated_answer: str) -> int:
    """
    Evaluate alignment of the LLM-generated answer with ground truth answer.

    Args:
        ground_truth_answer (str): Ground truth answer.
        generated_answer (str): LLM-generated answer.

    Returns:
        int: Alignment rating between 1 to 5.
    """
    prompt = f"""Evaluate the alignment of the following AI-generated answer with the ground truth answer.

    QUESTION: {query}

    GROUND TRUTH ANSWER: {ground_truth_answer}

    AI-GENERATED ANSWER: {generated_answer}

    Rate the alignment of the AI-generated answer with the ground truth answer on a scale of 1 to 5, where:
    1 = Completely unaligned (contains major factual errors or contradictions)
    2 = Mostly aligned (contains several factual errors or misinterpretations)
    3 = Partially aligned (contains minor factual errors while getting some things right)
    4 = Mostly aligned (accurately captures most information with minimal errors)
    5 = Completely aligned (fully aligned with the ground truth, no factual errors)

    Provide your rating and a brief explanation of your reasoning.

    Return your answer as a JSON object with two keys namely "rating" (integer) and "reasoning"
    """
    response = gemini_client.models.generate_content(
        model=LLM,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.0, responseMimeType="application/json"
        ),
    )
    result = json.loads(response.text)
    return result["rating"]

def eval_model(model: str, eval_dataset: List[dict]) -> None:
    """
    Evaluate retrieval and generation performance of Voyage AI and CLIP models.

    Args:
        model (str): Model to evaluate, one of "voyage" or "clip".
        eval_dataset (List[dict]): Evaluation dataset containing questions, ground truth answers, and relevant page numbers.
    """
    logger.debug(f"Model: {model}")
    rr = 0
    recall_at_5 = 0
    alignment = 0
    len_dataset = len(eval_dataset)
    for item in tqdm(eval_dataset):
        query = item["question"]
        answer = item["answer"]
        relevant_page_nums = item["page_numbers"]
        retrieved_keys = vector_search(query, model, display_images=False)
        retrieved_page_nums = [get_page_num(key) for key in retrieved_keys]
        rr += calculate_rr(relevant_page_nums, retrieved_page_nums)
        recall_at_5 += recall_at_k(relevant_page_nums, retrieved_page_nums, 5)
        generated_answer = generate_answer(query, model)
        alignment += eval_alignment(query, answer, generated_answer)
    logger.debug(f"MRR: {rr/len_dataset}")
    logger.debug(f"Avg. Recall @5: {recall_at_5/len_dataset}")
    logger.debug(f"Avg. Alignment: {alignment/len_dataset}")

eval_model("voyage", eval_dataset[:5])

eval_model("clip", eval_dataset[:5])

eval_model("voyage", eval_dataset[5:])

eval_model("clip", eval_dataset[5:])

eval_model("voyage", eval_dataset)

eval_model("clip", eval_dataset)

logger.info("\n\n[DONE]", bright=True)