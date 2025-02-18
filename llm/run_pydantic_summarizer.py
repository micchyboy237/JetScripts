from datetime import date
from decimal import Decimal
import json
from typing import Any, List, Optional, TypedDict

from jet.llm.ollama.base import Ollama
from jet.scrapers.utils import clean_text
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


class JobTypeEnum(str, Enum):
    FULL_TIME = "Full-Time"
    PART_TIME = "Part-Time"
    CONTRACT = "Contract"
    INTERNSHIP = "Internship"


class WorkScheduleEnum(str, Enum):
    FLEXIBLE = "Flexible"
    FIXED = "Fixed"
    SHIFT_BASED = "Shift-based"


class CurrencyEnum(str, Enum):
    USD = "$"
    PHP = "₱"
    EUR = "€"
    GBP = "£"
    INR = "₹"
    JPY = "¥"


class CountryEnum(str, Enum):
    USA = "United States"
    CANADA = "Canada"
    UK = "United Kingdom"
    GERMANY = "Germany"
    INDIA = "India"


class PaymentTermEnum(str, Enum):
    HOURLY = "Hourly"
    WEEKLY = "Weekly"
    BI_MONTHLY = "Bi-Monthly"  # Twice a month
    MONTHLY = "Monthly"


class Qualifications(BaseModel):
    mandatory: Optional[List[str]] = Field(
        ..., description="Required qualifications, skills, and experience")
    preferred: Optional[List[str]] = Field(
        ..., description="Preferred but not mandatory qualifications")


class TechStack(BaseModel):
    mandatory: Optional[List[str]
                        ] = Field(..., description="Mandatory tools, software, or platforms")
    preferred: Optional[List[str]] = Field(
        ..., description="Preferred but not mandatory tools, software, or platforms")


class WorkArrangement(BaseModel):
    schedule: Optional[WorkScheduleEnum] = Field(
        ..., description="Work schedule (e.g., Flexible, Fixed, Shift-based)")
    hoursPerWeek: Optional[int] = Field(...,
                                        description="Number of work hours per week")
    remote: Optional[bool] = Field(...,
                                   description="Indicates if remote work is allowed")


class SalaryRange(BaseModel):
    min: Optional[int] = Field(..., description="Minimum salary")
    max: Optional[int] = Field(..., description="Maximum salary")
    currency: Optional[str] = Field(
        ..., description="Currency of the salary (e.g., USD/hour, EUR/month, etc.)")


class Compensation(BaseModel):
    salaryRange: Optional[SalaryRange] = Field(
        ..., description="Salary range details")
    benefits: Optional[List[str]] = Field(
        ..., description="List of benefits (e.g., Health Insurance, Paid Time Off)")


class ApplicationProcess(BaseModel):
    applicationLinks: Optional[List[HttpUrl]] = Field(
        ..., description="List of URLs for application submission")
    contactInfo: Optional[List[str]] = Field(
        ..., description="List of recruiter or HR contact details")
    instructions: Optional[List[str]
                           ] = Field(..., description="List of instructions on how to apply")


class JobPosting(BaseModel):
    jobTitle: Optional[str] = Field(
        ..., description="Title of the job position")
    jobType: Optional[JobTypeEnum] = Field(
        ..., description="Type of employment")
    description: Optional[str] = Field(
        ..., description="Brief job summary")
    techStack: Optional[TechStack] = Field(
        ..., description="Job technological stack requirements")
    company: Optional[str] = Field(
        ..., description="Name of the hiring company or employer")
    country: Optional[CountryEnum] = Field(
        ..., description="Country where the job is located")
    compensation: Optional[Compensation] = Field(
        ..., description="Compensation details")
    collaboration: Optional[List[str]] = Field(
        ..., description="Teams or individuals the candidate will work with")
    workArrangement: Optional[WorkArrangement] = Field(
        ..., description="Work arrangement details")
    applicationProcess: Optional[ApplicationProcess] = Field(
        ..., description="Details about how to apply")
    postedDate: Optional[str] = Field(
        ..., description="Date when the job was posted")
    qualifications: Optional[Qualifications] = Field(
        ..., description="Job qualifications and requirements")
    responsibilities: Optional[List[str]
                               ] = Field(..., description="List of job responsibilities")


output_cls = JobPosting
output_sample = {
    "jobTitle": "Sample Job Title",
    "jobType": "Sample Job Type",
    "description": "Sample job description goes here.",
    "qualifications": {
        "mandatory": [
            "Sample mandatory qualification"
        ],
        "preferred": [
            "Sample preferred qualification"
        ]
    },
    "responsibilities": [
        "Sample responsibility"
    ],
    "company": "Sample Company",
    "industry": "Sample Industry",
    "location": {
        "city": "Sample City",
        "state": "Sample State",
        "country": "Sample Country",
        "remote": True
    },
    "skills": [
        "Sample Skill"
    ],
    "tools": [
        "Sample Tool"
    ],
    "collaboration": [
        "Sample Team"
    ],
    "workArrangement": {
        "schedule": "Sample Schedule",
        "hoursPerWeek": 40,
        "remote": True
    },
    "compensation": {
        "salaryRange": {
            "min": 50000,
            "max": 80000,
            "currency": "Sample Currency"
        },
        "benefits": [
            "Sample Benefit"
        ]
    },
    "applicationProcess": {
        "applicationLinks": [
            "https://www.samplecompany.com/apply"
        ],
        "contactInfo": [
            "hr@samplecompany.com"
        ],
        "instructions": [
            "Sample instructions on how to apply."
        ]
    },
    "postedDate": "2025-02-18"
}


sample_prompt = """
Job Summary: Link Building Specialist at Get Me Links

Company: Bronwyn Reynolds
Salary: $550/month
Job Type: Full-time (Remote)
Posted Date: February 18, 2025

Job Overview:
Get Me Links is hiring a Link Building Specialist with SEO experience. The role requires strong attention to detail, effective communication, and teamwork. Candidates must be ambitious, honest, and results-driven.

Requirements:
Experience: 2+ years as a link builder or SEO specialist
Skills: Fluent English (spoken & written), proficiency in Google Sheets/Excel, scripting (preferred but not required)
Work Conditions: Full-time only (no side gigs), remote work, flexible hours
Benefits:
Full social benefits after probation (SSS, PhilHealth, PAGIBIG)
13th-month pay (eligible after trial)
Paid holidays (national holidays + Christmas to New Year + 2 extra weeks)
Supportive and relaxed work environment with no micromanagement
Application Process:
Send your resume, two references, and a DISC personality test result to bronwyn@getmelinks.com
Use subject line: “Link Building Specialist for GetMeLinks - I’m [Your Name] and I’m an A player”
Complete a test task within 24 hours
If shortlisted, attend 1-2 interviews
Strict Application Rules: Only applications following instructions will be considered.
""".strip()

sample_response = """
{
  "jobTitle": "Link Building Specialist",
  "jobType": "Full-Time",
  "description": "Get Me Links is hiring a Link Building Specialist with SEO experience. The role requires strong attention to detail, effective communication, and teamwork. Candidates must be ambitious, honest, and results-driven.",
  "qualifications": {
    "mandatory": [
      "Fluent spoken and written English",
      "2+ years experience as a link builder or SEO specialist",
      "Proficiency in Excel/Google Sheets"
    ],
    "preferred": [
      "Script (Google Sheets) development skills"
    ]
  },
  "responsibilities": [
    "Capture order details from clients, ensuring clarity and accuracy",
    "Communicate effectively with the fulfillment team",
    "Ensure high-quality link-building processes",
    "Work collaboratively with the team to achieve goals"
  ],
  "company": "Get Me Links",
  "industry": "Digital Marketing",
  "location": {
    "remote": true
  },
  "skills": [
    "SEO",
    "Link Building",
    "Google Sheets",
    "Communication",
    "Attention to Detail"
  ],
  "tools": [
    "Google Sheets",
    "Excel"
  ],
  "collaboration": [
    "Fulfillment Team",
    "Clients",
    "Management"
  ],
  "workArrangement": {
    "schedule": "Flexible",
    "hoursPerWeek": 40,
    "remote": true
  },
  "compensation": {
    "salaryRange": {
      "min": 550,
      "currency": "USD"
    },
    "benefits": [
      "Full Social Benefits Package after trial period",
      "13th-month payment",
      "Paid holidays (national holidays + Christmas to New Year week + 2 weeks per year)"
    ]
  },
  "applicationProcess": {
    "contactInfo": [
      "bronwyn@getmelinks.com"
    ],
    "instructions": [
      "Send your resume, two references, and a DISC personality test result",
      "Use the subject line: 'Link Building Specialist for GetMeLinks - I’m [Your Name] and I’m an A player'",
      "Complete a test task within 24 hours",
      "Shortlisted candidates will attend 1-2 interviews"
    ]
  },
  "postedDate": "2025-02-18"
}
""".strip()

DEFAULT_SAMPLE_OUTPUT = json.dumps(output_sample, indent=1)

DEFAULT_SAMPLE = f"""
Example:
```text
{sample_prompt}
```
Response:
```json
{sample_response}
```
<end>
""".strip()

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context_str}
---------------------

Given the context information and not prior knowledge, answer the query.

Query: {query_str}
Response:
"""


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
            output_cls=output_cls,
            summary_template=self.qa_prompt,
        )

    # def summarize(self, query: str, texts: List[str], llm_kwargs: Optional[dict[str, Any]] = None) -> output_cls:
    #     return self.summarizer.get_response(query, texts, sample_str=DEFAULT_SAMPLE_OUTPUT, llm_kwargs=llm_kwargs)

    def summarize(self, query: str, contexts: str | List[str], output_cls: BaseModel, llm: Ollama) -> BaseModel:
        prompt_tmpl = PromptTemplate(PROMPT_TEMPLATE)

        schema = output_cls.model_json_schema()
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"

        if isinstance(contexts, str):
            contexts = [contexts]

        results: list[BaseModel] = []
        for context in contexts:
            response = llm.structured_predict(
                output_cls=output_cls,
                prompt=prompt_tmpl,
                context_str=context,
                # prompt_str=prompt,
                # schema_str=schema,
                # sample_str=DEFAULT_SAMPLE_OUTPUT,
                query_str=query,
                llm_kwargs={
                    "options": {"temperature": 0},
                    # "max_prediction_ratio": 0.5
                },
            )
            results.append(response)

        # Temporary single return
        # Should merge results into one object
        return results[0]


def run_extract_jobs():
    from jet.executor.command import run_command
    from jet.file import load_file
    from jet.file.utils import save_file
    from jet.logger import logger
    import json

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
    from jet.file import load_file
    from jet.file.utils import save_file
    from jet.logger import logger
    import json

    python_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/run_clean_jobs.py"
    command = f"python {python_file}"

    for line in run_command(command):
        if line.startswith("error: "):
            message = line[len("error: "):-2]
            logger.error(message)
        else:
            message = line[len("data: "):-2]
            logger.success(message)


class VectorNode(TypedDict):
    id: str
    score: float
    text: str
    metadata: Optional[dict]


class SearchNodesResponse(TypedDict):
    count: int
    data: list[VectorNode]


def search_nodes(query: str) -> SearchNodesResponse:
    from jet.memory.httpx import HttpxClient
    from fastapi import HTTPException
    import httpx
    from jet.logger import logger

    url = "http://0.0.0.0:8002/api/v1/rag/nodes"

    request_data = {
        "query": query,
        "rag_dir": "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json",
        "extensions": [
            ".md",
            ".mdx",
            ".rst"
        ],
        "json_attributes": [
            "title",
            "tags",
            "details"
        ],
        "exclude_json_attributes": [
            "overview"
        ],
        "metadata_attributes": [
            "id",
            "title",
            "link",
            "company",
            "posted_date",
            "salary",
            "job_type",
            "domain",
            "location",
            "entities"
        ],
        "system": "You are a job applicant providing tailored responses during an interview.\n  Always answer questions using the provided context as if it is your resume, \n  and avoid referencing the context directly.\n  Some rules to follow:\n  1. Never directly mention the context or say 'According to my resume' or similar phrases.\n  2. Provide responses as if you are the individual described in the context, focusing on professionalism and relevance.",
        "chunk_size": 1024,
        "chunk_overlap": 40,
        "sub_chunk_sizes": [
            512,
            256,
            128
        ],
        "with_hierarchy": False,
        "top_k": None,
        "model": "llama3.2",
        "embed_model": "mxbai-embed-large",
        "mode": "fusion",
        "store_path": "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/.cache/deeplake/store_1",
        "score_threshold": 0.7,
        "split_mode": [],
        "fusion_mode": "simple"
    }

    try:
        client = HttpxClient()
        response = client.post(url, json=request_data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {
                     e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code, detail=e.response.text
        ) from e  # Raising the original error as context
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e  # Re-raise the original exception


def main():
    run_extract_jobs()
    run_clean_jobs()

    keyword = "React Native"
    search_result = search_nodes(keyword)
    # Extracting search_data from search_result
    search_data = search_result["data"]
    # Creating a set of valid IDs for fast lookup
    search_data_ids = [s_item['metadata']['id'] for s_item in search_data]

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
    output_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/job-postings.json"
    data = load_file(data_file) or []
    results = load_file(output_file) or []

    # Extracting the set of existing IDs in the results
    existing_ids = [item['id'] for item in results]

    # Filtering data to include only items whose 'id' exists in search_data_ids
    data = [item for id in search_data_ids for item in data if item['id'] == id]

    # Filtering data to include only items whose ID is not already in results
    data = [item for item in data if item['id'] not in existing_ids]

    json_attributes = [
        # "id",
        # "link",
        "title",
        "company",
        "posted_date",
        "salary",
        "job_type",
        # "overview",
        "tags",
        "domain",
        "location",
        "details",
        # "entities",
    ]

    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    job_postings = []
    data_chunks = []
    for item in data:
        text_parts = [f"{attr.title().replace('_', ' ')}: {str(item[attr])}"
                      for attr in json_attributes
                      if attr in item and item[attr]]
        text_content = "\n".join(text_parts) if text_parts else ""
        cleaned_text_content = clean_text(text_content)
        text_chunks = splitter.split_text(cleaned_text_content)
        data_chunks.append(text_chunks)

    query = 'Extract complete relevant job post information that can be derived from the context information. Use null if not available in context.'
    for idx, text_chunks in enumerate(tqdm(data_chunks, total=len(data_chunks), unit="chunk")):
        # Summarize
        summarizer = Summarizer(llm=llm)

        try:
            response = summarizer.summarize(
                query, text_chunks, output_cls, llm)
        except Exception as e:
            logger.error(e)
            continue

        jobId = data[idx]['id']
        jobLink = data[idx]['link']
        result = {
            "id": jobId,
            "link": jobLink,
            **response.__dict__
        }
        job_postings.append(result)

        # Inspect response
        logger.success(format_json(result))

        save_file(job_postings, output_file)


if __name__ == "__main__":
    main()
