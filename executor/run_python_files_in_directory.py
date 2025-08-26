from jet.executor.python_runner import run_python_files_in_directory

if __name__ == "__main__":
    includes = []
    excludes = []
    run_python_files_in_directory(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/GenAI_Agents/converted-notebooks/all_agents_tutorials",
        includes=includes,
        excludes=excludes)
