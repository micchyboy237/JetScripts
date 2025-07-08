from typing import List, Dict, Any
from jet.data.stratified_sampler import ProcessedDataString, StratifiedSampler
from jet.file.utils import load_file

original_chunks_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/docs.json"
merged_docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/merged_docs.json"

# Load files with error handling
try:
    original_chunks: List[Dict[str, Any]] = load_file(original_chunks_file)
    merged_docs: List[Dict[str, Any]] = load_file(merged_docs_file)
except Exception as e:
    raise ValueError(f"Failed to load input files: {e}")

if not original_chunks or not merged_docs:
    raise ValueError("Input files are empty or invalid")

# Create dictionary with error handling for missing doc_ids
original_chunks_dict = {chunk["doc_id"]: chunk for chunk in original_chunks}

for merged_doc in merged_docs:
    merged_original_doc_ids: List[str] = merged_doc.get("original_doc_ids", [])
    merged_original_docs: List[Dict[str, Any]] = []
    for doc_id in merged_original_doc_ids:
        if doc_id not in original_chunks_dict:
            print(
                f"Warning: doc_id {doc_id} not found in original_chunks_dict")
            continue
        merged_original_docs.append(original_chunks_dict[doc_id])

    # Print headers
    print(f"Merged chunk header: {merged_doc.get('header', 'No header')}")
    print("Original doc headers in this merged chunk:")

    # Create ProcessedDataString with correct source type (string)
    data: List[ProcessedDataString] = [
        ProcessedDataString(
            source=doc["content"],
            category_values=[
                doc.get("url", ""),
                merged_doc.get("parent_header", "") or "",
                merged_doc.get("header", "")
            ]
        )
        for doc in merged_original_docs
    ]

    if not data:
        print("No valid data for sampling in this merged chunk")
        continue

    sampler = StratifiedSampler(data, num_samples=1)
    sampled = sampler.get_samples()
    print("Sampled strings:")
    for s in sampled:
        print(f"Source: {s['source']}")
        print(f"Category Values: {s['category_values']}")
    print()
