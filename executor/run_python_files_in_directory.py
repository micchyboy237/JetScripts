from jet.executor.python_runner import run_python_files_in_directory

if __name__ == "__main__":
    includes = ["*21_*", "*best_*",]
    excludes = ["*18_*"]
    run_python_files_in_directory(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques",
        includes=includes,
        excludes=excludes)
