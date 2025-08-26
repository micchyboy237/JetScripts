import os
import shutil
from jet.executor.python_runner import run_python_files_in_directory

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    includes = []
    excludes = []
    run_python_files_in_directory(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/GenAI_Agents/converted-notebooks/all_agents_tutorials",
        includes=includes,
        excludes=excludes,
        output_dir=OUTPUT_DIR,
    )
