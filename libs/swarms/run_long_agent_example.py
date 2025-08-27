import os
from pathlib import Path
from typing import List, Union

from jet.libs.swarms.examples.multi_agent.paper_implementations.long_agent import LongAgent

DATA_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"


def get_all_files(data_path: Union[str, Path]) -> List[Path]:
    """
    Recursively collect all supported files (.pdf, .md, .txt) from the given directory.

    Args:
        data_path (Union[str, Path]): The root directory to search for files

    Returns:
        List[Path]: List of file paths with supported extensions
    """
    supported_extensions = {'.pdf', '.md', '.txt'}
    file_paths = []

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_path}")

    for root, _, files in os.walk(data_path):
        for file in files:
            if Path(file).suffix.lower() in supported_extensions:
                file_paths.append(Path(root) / file)

    return file_paths


def main():
    """
    Main function to process all documents under DATA_PATH using LongAgent.
    """
    try:
        # Initialize LongAgent
        agent = LongAgent(
            name="DocumentProcessor",
            description="Processes all documents in the specified directory",
            token_count_per_agent=16000,
            output_type="final",
            model_name="ollama/llama3.2",
            aggregator_model_name="ollama/llama3.2"
        )

        # Get all supported files
        file_paths = get_all_files(DATA_PATH)

        if not file_paths:
            print(f"No supported files found in {DATA_PATH}")
            return

        print(f"Found {len(file_paths)} files to process: {file_paths}")

        # Process all files and generate report
        final_report = agent.run(file_paths)

        print("\nFinal Report:")
        print(final_report)

    except Exception as e:
        print(f"Error processing documents: {str(e)}")


if __name__ == "__main__":
    main()
