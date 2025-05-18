import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import TreeSummarize
from jet.vectors import SettingsManager, SettingsDict, QueryProcessor
from jet.file import save_json


class DataLoader:
    def __init__(self):
        self.reader = SimpleDirectoryReader(
            input_files=["./data/paul_graham/paul_graham_essay.txt"]
        )

    def load_data(self):
        """Load data from the downloaded files"""
        docs = self.reader.load_data(show_progress=True)
        return docs


class Summarizer:
    def __init__(self):
        settings = SettingsDict(
            llm_model="llama3.1",
            embedding_model="nomic-embed-text",
            chunk_size=512,
            chunk_overlap=50,
            base_url="http://localhost:11434",
        )
        settings_manager = SettingsManager.create(settings)
        self.summarizer = TreeSummarize(
            llm=settings_manager.llm,
            streaming=True,
            verbose=True,
        )

    def summarize(self, question: str, text: str) -> str:
        """Summarize the provided text based on the given question"""
        response = self.summarizer.get_response(question, [text])
        return response


def main() -> None:

    loader = DataLoader()
    docs = loader.load_data()
    text = docs[0].text

    summarizer = Summarizer()
    question = "who is Paul Graham?"
    summary = summarizer.summarize(question, text)
    print(summary)


if __name__ == "__main__":
    main()

# Installation Instructions
# If you're using Google Colab, run the following command to install LlamaIndex:
# `!pip install llama-index`
