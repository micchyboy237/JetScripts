from jet.code.extraction.extract_notebook_texts import run_text_extraction

if __name__ == "__main__":
    input_path = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/swarms"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/swarms/notebook_texts"

    run_text_extraction(input_path, output_dir)
