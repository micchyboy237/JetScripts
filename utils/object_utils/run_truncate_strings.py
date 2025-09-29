from jet.file.utils import load_file, save_file
from jet.utils.object import truncate_strings
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    json_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/tavilly/generated/run_tavily_search/result.json"
    json_data = load_file(json_path)
    result = truncate_strings(json_data, 150, "...")
    save_file(result, f"{OUTPUT_DIR}/tavily_search_truncated_result.json")
