from jet.file.utils import load_file, save_file
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard


if __name__ == "__main__":
    eval_model = "gemma3:4b"
    query = "What are the steps in registering a National ID in the Philippines?"

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_llm_reranker/nodes_with_scores.json"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/semantic_search/eval/generated/run_eval_context_relevancy"
    data = load_file(data_file)
    nodes_with_scores = [d["node"] for d in data["results"]]

    top_k = 5
    # contexts = [
    #     node.text
    #     for node in nodes_with_scores[:top_k]
    # ]
    contexts = [
        "# How to Get Philippine National ID (PhiLID Application Requirements)  \nLast Updated on: January 6, 2025 by Gabriel Spencer  \nEverything you need to know about how to get a Philippine National ID in 2025, the complete requirements, application fees and registration of PhilID.  \nGreat news! All Filipino citizens and foreign residents in the Philippines will be required to have a National ID (PhilID). The Philippine Identification System (PhilSys) Act's primary goal is to establish a single national ID card for all citizens and foreign residents of the Philippines.  \nThe Philippine Statistics Authority (PSA) started the National ID pre-registration process a few years ago. The PSA began collecting information and records for low-income households so that they can have faster digital access to financial assistance and government services.  \nThe PSA targets majority of Filipinos registered for the Philippine National ID system. If you want to prepare for it, here's everything you need to know about getting a PhilID.  \nWe're sharing here how to get a National ID in the Philippines, the step-by-step guides, and procedures. You should also take note of the complete requirements in getting a Philippine National ID in 2025.  \nTable of Contents:  \nToggle  \n* What is Philippine National ID?\n* Why Should You Get a PhilID?\n* 5 Benefits of Philippine National ID Card (PhilID)\n* 1. Uniformity\n* 2. Quick Access\n* 3. Faster Transactions\n* 4. Eliminates Fraud\n* 5. Transparency\n* Philippine National ID Card Requirements:\n* Philippine National ID Requirements for Resident Aliens:\n* How to Get a National ID in the Philippines in 2025:\n* How to Get a PhilID (Philippine National ID) - Application Guides (with Pictures)\n* 1. Prepare the documents required for PhilID\n* 2. Go to any PhilID Registration Center\n* 3. Fill out your Philippine National ID Application Form\n* 4. Have your biometrics information captured\n* 5. Generate a PSN (PhilSys Number)\n* 6. Receive your National ID Card\n* What information does Philippine National ID contain?\n* PhilID Application Requirements for Minors\n* PhilID Application Special Arrangements\n* How much is the cost of PhilID registration?\n* Where can we use our PhilID and PSN?\n* What is the validity and expiration of Philippine National ID?\n* Will PhilID replace other government-issued ID?\n* What is Philippine National ID Number?\n* How to Apply for Philippine National ID Abroad?\n* How to Apply for Philippine National ID Online?"
    ]

    eval_result = evaluate_context_relevancy(eval_model, query, contexts)

    logger.success(format_json(eval_result))
    copy_to_clipboard(eval_result)

    output_file = f"{output_dir}/eval_result.json"
    save_file(eval_result, output_file)
