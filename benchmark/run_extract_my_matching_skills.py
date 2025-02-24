from jet.llm.ollama.base import Ollama
from jet.scrapers.utils import clean_text
from jet.utils.object import extract_values_by_paths
from llama_index.core.utils import set_global_tokenizer
from pydantic import BaseModel, Field

from jet.file.utils import save_file, load_file
from jet.token.token_utils import get_ollama_tokenizer
from jet.transformers.formatters import format_json
from jet.logger import logger

from llama_index.core import PromptTemplate
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.response_synthesizers import TreeSummarize
from tqdm import tqdm


class JobSkills(BaseModel):
    skills: list[str] = Field(...,
                              description="Relevant skills provided in query.")


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
            "Sample response:\n"
            "```json\n"
            "{sample_str}\n"
            "```\n"
            "\n"
            "Given the context information, sample response, schema and not prior knowledge, "
            "answer the query.\n"
            "The generated JSON must pass the provided schema when validated.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

        self.summarizer = TreeSummarize(
            llm=self.llm,
            verbose=self.verbose,
            streaming=self.streaming,
            output_cls=JobSkills,
            summary_template=self.qa_prompt,
        )

    def summarize(self, query: str, contexts: list[str]) -> JobSkills:
        results = []
        for context in contexts:
            response = self.llm.structured_predict(
                output_cls=JobSkills,
                prompt=self.qa_prompt,
                context_str=context,
                query_str=query,
                llm_kwargs={"options": {"temperature": 0}},
            )
            results.append(response)
        return results[0]


def run_extract_jobs():
    from jet.executor.command import run_command
    python_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/job-scraper.py"
    command = f"python {python_file}"

    for line in run_command(command):
        if line.startswith("error: "):
            message = line[len("error: "):-2]
            logger.error(message)
        else:
            message = line[len("data: "):-2]
            logger.success(message)


def run_clean_jobs():
    from jet.executor.command import run_command
    python_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/run_clean_jobs.py"
    command = f"python {python_file}"

    for line in run_command(command):
        if line.startswith("error: "):
            message = line[len("error: "):-2]
            logger.error(message)
        else:
            message = line[len("data: "):-2]
            logger.success(message)


def main():
    # Settings initialization
    model = "llama3.1"
    chunk_size = 1024
    chunk_overlap = 128

    output_cls = JobSkills
    my_skills = [
        "React.js",
        "React Native",
        "Node.js",
        "Python",
        "PostgreSQL",
        "MongoDB",
        "Firebase",
        "AWS",
    ]
    query = f'Analyze the job post information to determine all skills mentioned or equivalent that can be derived from context. Use these as basis: {", ".join([f'"{skill}"' for skill in my_skills])}.'

    tokenizer = get_ollama_tokenizer(model)
    set_global_tokenizer(tokenizer)

    llm = Ollama(
        model=model,
    )

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    output_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/job-matched-skills.json"
    data = load_file(data_file) or []
    results = load_file(output_file) or []

    # Extracting the set of existing IDs in the results
    existing_ids = [item['id'] for item in results]

    # Filtering data to include only items whose ID is not already in results
    data = [item for item in data if item['id'] not in existing_ids]

    json_attributes = [
        "title",
        "details",
    ]

    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    job_postings = []
    data_chunks = []
    for item in data:
        json_parts_dict = extract_values_by_paths(
            item, json_attributes, is_flattened=True)
        text_parts = []
        for key, value in json_parts_dict.items():
            value_str = str(value)
            if isinstance(value, list):
                value_str = ", ".join(value)
            text_parts.append(
                f"{key.title().replace('_', ' ')}: {value_str}")
        text_content = "\n".join(text_parts) if text_parts else ""
        cleaned_text_content = clean_text(text_content)
        text_chunks = splitter.split_text(cleaned_text_content)
        data_chunks.append(text_chunks)

    for idx, text_chunks in enumerate(tqdm(data_chunks, total=len(data_chunks), unit="chunk")):
        # Summarize
        summarizer = Summarizer(llm=llm)

        try:
            response = summarizer.summarize(
                query, text_chunks, output_cls, llm)
        except Exception as e:
            logger.error(e)
            continue

        job_id = data[idx]['id']
        job_link = data[idx]['link']

        matched_skills = [
            my_skill for my_skill in my_skills
            if any(my_skill in skill for skill in response.skills)
        ]
        result = {
            "id": job_id,
            "link": job_link,
            "text": "\n\n".join(text_chunks),
            "matched_skills": matched_skills,
        }
        job_postings.append(result)

        # Inspect response
        logger.success(format_json(result))

        save_file(job_postings, output_file)


if __name__ == "__main__":
    main()
