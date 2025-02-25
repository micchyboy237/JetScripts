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
Hi Benson,

I'm interested in the position as the tech stack mentioned seems to be an ideal fit for my skillset.

My primary skills includes using React, TypeScript and Node.js

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
Prompt:
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
Given the prompt, schema and not prior knowledge, answer the query.
The generated JSON must pass the provided schema when validated.
Query: {query_str}

Prompt:
```text
{prompt_str}
```
Response:
"""

MESSAGE_TEMPLATE = """
Hi <employer_name>,

I'm interested in the position as the tech stack mentioned seems to be an ideal fit for my skillset.
My primary skills <continue>.

<other_relevant_info>

Here is a link to my website with portfolio and latest resume:
https://jethro-estrada.web.app
"""

DEFAULT_QUERY = f"""
Generate a cover letter based on provided job post information. A company may be an organization or employer name. Follow instructions relevant to the subject or message if any.

You may follow this message template:
```template
{MESSAGE_TEMPLATE}
```
""".strip()


class Summarizer:
    def __init__(self, llm, verbose: bool = True, streaming: bool = False, prompt_tmpl: str = PROMPT_TEMPLATE):
        self.llm = llm
        self.verbose = verbose
        self.streaming = streaming

        self.qa_prompt = PromptTemplate(prompt_tmpl)

    def summarize(self, query: str, prompt: str, output_cls: BaseModel, llm: Ollama) -> BaseModel:
        response = llm.structured_predict(
            output_cls=output_cls,
            prompt=self.qa_prompt,
            # context_str=context,
            # schema_str=schema,
            sample_str=DEFAULT_SAMPLE,
            query_str=query,
            prompt_str=prompt,
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

    query = DEFAULT_QUERY

    # Extracting the set of existing IDs in the results
    existing_ids = [item['id'] for item in results]

    # Filtering data to include only items whose ID is not already in results
    data = [item for item in data if item['id'] not in existing_ids]

    json_attributes = [
        "title",
        "company",
        "salary",
        "job_type",
        "hours_per_week",
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
        # for key, value in json_parts_dict.items():
        #     value_str = str(value)
        #     if isinstance(value, list):
        #         value_str = ", ".join(value)
        #     text_parts.append(
        #         f"{key.title().replace('_', ' ')}: {value_str}")

        optional_attrs = [
            "salary",
            "job_type",
            "hours_per_week",
        ]

        text_parts = [
            f"Job Title: {json_parts_dict["title"]}",
            f"Company or Employer: {json_parts_dict["company"]}",
        ]

        for key in optional_attrs:
            if json_parts_dict.get(key):
                prefix = key.title().replace("_", " ")
                text_parts.append(f"{prefix}: {json_parts_dict[key]}")

        text_parts.append(f"Job Details:\n{json_parts_dict["details"]}",)

        text_content = "\n".join(text_parts) if text_parts else ""
        cleaned_text_content = clean_text(text_content)
        text_chunks = splitter.split_text(cleaned_text_content)
        data_chunks.append(text_chunks[0])

    summarizer = Summarizer(llm=llm)

    for idx, text_chunk in enumerate(tqdm(data_chunks, total=len(data_chunks), unit="chunk")):
        # Summarize
        prompt = text_chunk

        try:
            response = summarizer.summarize(
                query, prompt, output_cls, llm)
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
