from typing import Optional, List
from pydantic import BaseModel
from jet.logger import logger
from jet.vectors.rag import SettingsManager
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.types import PydanticProgramMode
from llama_index.core import SimpleDirectoryReader, PromptTemplate


class AnimeDetails(BaseModel):
    seasons: int
    episodes: int
    additional_info: Optional[str] = None


class Summarizer:
    def __init__(self, llm, verbose: bool = True, streaming: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.streaming = streaming

        self.qa_prompt = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information, schema and not prior knowledge, "
            "answer the query.\n"
            "The generated JSON must pass the provided schema when validated.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

        self.refine_prompt = PromptTemplate(
            "The original query is as follows: {query_str}\n"
            "We have provided an existing answer: {existing_answer}\n"
            "We have the opportunity to refine the existing answer "
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the query. "
            "The generated JSON must pass the provided schema when validated.\n"
            "If the context isn't useful, return the original answer.\n"
            "Refined Answer: "
        )

        self.summarizer = TreeSummarize(
            llm=self.llm,
            verbose=self.verbose,
            streaming=self.streaming,
            output_cls=AnimeDetails,
            summary_template=self.qa_prompt,
        )

    def summarize(self, question: str, texts: List[str]) -> AnimeDetails:
        return self.summarizer.get_response(question, texts)


def main():
    # Settings initialization
    settings_manager = SettingsManager.create()
    settings_manager.pydantic_program_mode = PydanticProgramMode.LLM

    # Load data
    reader = SimpleDirectoryReader(
        input_files=[
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/scraped_urls/www_imdb_com_title_tt32812118.md"]
    )
    docs = reader.load_data()
    texts = [doc.text for doc in docs]

    # Summarize
    summarizer = Summarizer(llm=settings_manager.llm)
    question = 'How many seasons and episodes does "I’ll Become a Villainess Who Goes Down in History" anime have?'
    response = summarizer.summarize(question, texts)

    # Inspect response
    logger.success(response)
    logger.success(f"Seasons: {response.seasons}")
    logger.success(f"Episodes: {response.episodes}")
    logger.success(f"Additional Info: {response.additional_info}")


if __name__ == "__main__":
    main()
