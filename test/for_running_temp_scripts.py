import os
from jet.logger import logger
from llama_index.core.readers.file.base import SimpleDirectoryReader

if __name__ == "__main__":
    input_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    output_dir = os.path.join(os.path.dirname(input_dir), "results")
    os.makedirs(output_dir, exist_ok=True)

    # Configure the MarkdownReader to retain headings
    # markdown_reader = MarkdownReader(
    #     preserve_headers=True,
    #     remove_hyperlinks=False,
    #     # remove_images=False,
    # )

    # Use SimpleDirectoryReader with the custom MarkdownReader
    documents = SimpleDirectoryReader(
        input_dir,
        required_exts=[".md"],
        # file_extractor={".md": markdown_reader}
    ).load_data()

    texts = [doc.text for doc in documents]

    combined_file_path = os.path.join(output_dir, "combined.txt")
    with open(combined_file_path, "w") as f:
        f.write("\n\n\n".join(texts))
    logger.success("Results saved to:", combined_file_path)
