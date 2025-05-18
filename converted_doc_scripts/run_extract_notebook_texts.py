from jet.code.extraction.extract_notebook_texts import run_text_extraction

if __name__ == "__main__":
    input_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/notebooks"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/docs_text_only"

    run_text_extraction(input_path, output_dir)
