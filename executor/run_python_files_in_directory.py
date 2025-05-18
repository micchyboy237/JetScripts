from jet.executor.python_runner import run_python_files_in_directory

if __name__ == "__main__":
    excludes = ["*1_*", "*2_*", "*3_*", "*4_*", "*5_*", "*6_*"]
    run_python_files_in_directory(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques", excludes=excludes)
