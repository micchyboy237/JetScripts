import os
import shutil

from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_jobs_ai_llm_python
from jet.logger import logger
from jet.wordnet.analyzers.analyze_subject import analyze_subjects

# Define file paths
docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
output_dir = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(output_dir, ignore_errors=True)

# Load texts
texts = load_sample_jobs_ai_llm_python()

# Analyze subjects
results = analyze_subjects(texts)

# Save results
logger.info(f"Results: {len(results)}")
os.makedirs(output_dir, exist_ok=True)
save_file(results, f"{output_dir}/results.json")
