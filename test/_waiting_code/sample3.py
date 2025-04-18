import os
from jet.file.utils import load_file, save_file
from transformers import pipeline
import torch

# Initialize pipeline with MiniLM
qa_pipeline = pipeline(
    "question-answering",
    # model="huawei-noah/TinyBERT_General_4L_312D",
    model="microsoft/MiniLM-L12-H384-uncased",
    # model="google/mobilebert-uncased",
    # model="distilroberta-base",
    device=torch.device("mps")  # Use M1 GPU
)

data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/generated/search/top_anime_romantic_comedy_reddit_2024-2025/top_context_nodes.json"
data = load_file(data_file)

# Question
question = f"Does the text contain answers to the query? Query: \"{data["query"]}\""

# Filter results
filtered_results = []
for result in data["results"]:
    text = result["text"]
    answer = qa_pipeline(question=question, context=text)
    result["answer"] = answer

    if answer["score"] > 0.5 and answer["answer"].strip():
        filtered_results.append(result)

# Output
for result in filtered_results:
    print(result["text"])


output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
save_file({
    "all_count": len(data["results"]),
    "all_results": data["results"],
}, os.path.join(
    output_dir, "all_results.json"))
save_file({
    "filtered_results_count": len(filtered_results),
    "filtered_results": filtered_results,
}, os.path.join(
    output_dir, "filtered_results.json"))
