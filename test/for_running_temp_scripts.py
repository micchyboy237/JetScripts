from datetime import date
from decimal import Decimal
import json
from typing import Any, List, Optional, TypedDict

from jet.llm.ollama.base import Ollama
from jet.scrapers.utils import clean_text
from jet.utils.object import extract_values_by_paths
from jsonschema.exceptions import ValidationError
from llama_index.core.utils import set_global_tokenizer
from pydantic import BaseModel, EmailStr, HttpUrl, Field
from enum import Enum

from jet.file.utils import save_file, load_file
from jet.token.token_utils import get_ollama_tokenizer
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.vectors import SettingsManager

from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.types import PydanticProgramMode
from tqdm import tqdm


from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class JobCoverLetter(BaseModel):
    subject: str
    message: str


output_cls = JobCoverLetter

prompt_sample = """
Job Title: React native engineer
Job Details:
Responsibilities:
Build and maintain mobile applications with React Native.
Manage complex state and handle large, dynamic datasets.
Integrate and optimise APIs for seamless functionality.
Collaborate with teams to deliver innovative features.
Requirements:
Strong experience with React Native and React
Proficient in state management (e.g., Redux, Zustand, or Context API).
Skilled in handling large datasets and dynamic data.
Expertise in API integration and optimisation.
Interest in AI technologies.
Nice-to-Have:
Familiarity with Docker and containerisation.
Experience with Convex
Understanding of AWS or other cloud platforms.
What We Offer:
Opportunity to work on AI-driven projects.
Challenging and innovative work environment.
Flexible and remote-friendly options.
""".strip()
subject_sample = "Application for React Native Engineer Position"
message_sample = """
I am excited to apply for the React Native Engineer position. With extensive experience in mobile app development using React Native and React, I believe my skills align perfectly with your requirements.

I have strong expertise in state management solutions like Redux, Zustand and Context API, as well as significant experience handling large datasets and optimizing API integrations. I'm also familiar with Docker containerization and AWS cloud platforms.

I'm particularly interested in working on AI-driven projects and contributing to an innovative environment. My collaborative nature and technical skills would allow me to effectively build and maintain mobile applications while working with your team.

Here is a link to my website with portfolio and latest resume:
https://jethro-estrada.web.app

I look forward to discussing how I can contribute to your projects.

Best regards
""".strip()
output_sample = {
    "subject": subject_sample,
    "message": message_sample,
}


DEFAULT_SAMPLE = f"""
Example:
```text
{prompt_sample}
```
Response:
```json
{json.dumps(output_sample, indent=1)}
```
<end>
""".strip()

PROMPT_TEMPLATE = """\
Sample:
{sample_str}

Context information is below.
---------------------
{context_str}
---------------------

Given the context information, sample, schema and not prior knowledge, answer the query.
The generated JSON must pass the provided schema when validated.
Query: {query_str}
```text
{prompt_str}
```
Response:
"""


class Summarizer:
    def __init__(self, llm, verbose: bool = True, streaming: bool = False, prompt_tmpl: str = PROMPT_TEMPLATE):
        self.llm = llm
        self.verbose = verbose
        self.streaming = streaming

        self.qa_prompt = PromptTemplate(prompt_tmpl)

    def summarize(self, query: str, prompt: str, context: str, output_cls: BaseModel, llm: Ollama) -> BaseModel:
        response = llm.structured_predict(
            output_cls=output_cls,
            prompt=self.qa_prompt,
            context_str=context,
            prompt_str=prompt,
            # schema_str=schema,
            sample_str=DEFAULT_SAMPLE,
            query_str=query,
            llm_kwargs={
                "options": {"temperature": 0},
                # "max_prediction_ratio": 0.5
            },
        )

        return response


class VectorNode(TypedDict):
    id: str
    score: float
    text: str
    metadata: Optional[dict]


def main():
    # Settings initialization
    model = "llama3.1"
    chunk_size = 1024
    chunk_overlap = 128

    tokenizer = get_ollama_tokenizer(model)
    set_global_tokenizer(tokenizer)

    llm = Ollama(
        model=model,
    )

    # Load data
    # reader = SimpleDirectoryReader(
    #     input_files=[
    #         "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/scraped_urls/www_imdb_com_title_tt32812118.md"]
    # )
    # docs = reader.load_data()
    # texts = [doc.text for doc in docs]

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    output_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/job-cover-letters.json"
    data = load_file(data_file) or []
    results = load_file(output_file) or []

    query = 'Generate a cover letter based on provided job post information in context.'

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
        data_chunks.append(text_chunks[0])

    for idx, text_chunk in enumerate(tqdm(data_chunks, total=len(data_chunks), unit="chunk")):
        # Summarize
        summarizer = Summarizer(llm=llm)
        prompt = text_chunk
        context =

        try:
            response = summarizer.summarize(
                query, prompt, context, output_cls, llm)
        except Exception as e:
            logger.error(e)
            continue

        job_id = data[idx]['id']
        job_link = data[idx]['link']

        result = {
            "id": job_id,
            "link": job_link,
            "text": text_chunk,
            "response": response.__dict__,
        }
        job_postings.append(result)

        # Inspect response
        logger.success(format_json(result))

        save_file(job_postings, output_file)


if __name__ == "__main__":
    main()
