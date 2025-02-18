from datetime import date
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, EmailStr, HttpUrl, Field

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


class Location(BaseModel):
    city: Optional[str] = Field(
        None, description="City where the job is located")
    state: Optional[str] = Field(
        None, description="State where the job is located")
    country: Optional[str] = Field(...,
                                   description="Country where the job is located")
    remote: Optional[bool] = Field(
        None, description="Indicates if remote work is allowed")


class Qualifications(BaseModel):
    mandatory: Optional[List[str]] = Field(
        ..., description="Required qualifications, skills, and experience")
    preferred: Optional[List[str]] = Field(
        None, description="Preferred but not mandatory qualifications")


class WorkArrangement(BaseModel):
    schedule: Optional[str] = Field(
        None, description="Work schedule (e.g., Flexible, Fixed, Shift-based)")
    hoursPerWeek: Optional[int] = Field(
        None, description="Number of work hours per week")
    remote: Optional[bool] = Field(
        None, description="Indicates if remote work is allowed")


class SalaryRange(BaseModel):
    min: Optional[int] = Field(None, description="Minimum salary")
    max: Optional[int] = Field(None, description="Maximum salary")
    currency: Optional[str] = Field(...,
                                    description="Currency of the salary (e.g., USD, EUR)")


class Compensation(BaseModel):
    salaryRange: Optional[SalaryRange] = Field(
        None, description="Salary range details")
    benefits: Optional[List[str]] = Field(
        None, description="List of benefits (e.g., Health Insurance, Paid Time Off)")


class ApplicationProcess(BaseModel):
    applicationLinks: Optional[List[HttpUrl]] = Field(
        None, description="List of URLs for application submission")
    contactInfo: Optional[List[str]] = Field(
        None, description="List of recruiter or HR contact details")
    instructions: Optional[List[str]] = Field(
        None, description="List of instructions on how to apply")


class JobPosting(BaseModel):
    jobTitle: str = Field(..., description="Title of the job position")
    jobType: str = Field(
        ..., description="Type of employment (e.g., Full-Time, Part-Time, Contract, Internship)")
    description: str = Field(..., description="Brief job summary")
    qualifications: Qualifications = Field(
        ..., description="Job qualifications and requirements")
    responsibilities: Optional[List[str]] = Field(...,
                                                  description="List of job responsibilities")
    company: Optional[str] = Field(...,
                                   description="Name of the hiring company or employer")
    industry: Optional[str] = Field(
        ..., description="Industry related to the job (e.g., Technology, Healthcare, Finance)")
    location: Optional[Location] = Field(...,
                                         description="Job location details")
    skills: Optional[List[str]] = Field(
        None, description="Required technical and soft skills")
    tools: Optional[List[str]] = Field(
        None, description="List of required tools, software, or platforms")
    collaboration: Optional[List[str]] = Field(
        None, description="Teams or individuals the candidate will work with")
    workArrangement: Optional[WorkArrangement] = Field(
        None, description="Work arrangement details")
    compensation: Optional[Compensation] = Field(
        None, description="Compensation details")
    applicationProcess: Optional[ApplicationProcess] = Field(
        None, description="Details about how to apply")
    postedDate: date = Field(default_factory=date.today,
                             description="Date when the job was posted")

    # class Config:
    #     orm_mode = True


output_cls = JobPosting


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
            "The generated answer must follow the provided schema.\n"
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
            "The generated answer must follow the provided schema.\n"
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

    def summarize(self, question: str, texts: List[str]) -> output_cls:
        return self.summarizer.get_response(question, texts)


def main():
    # Settings initialization
    model = "llama3.1"
    chunk_size = 1024
    chunk_overlap = 200
    # embedding_model = "mxbai-embed-large"
    settings_manager = SettingsManager.create(settings={
        "llm_model": model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        # "embedding_model": embedding_model,
    })
    settings_manager.pydantic_program_mode = PydanticProgramMode.LLM

    # Load data
    # reader = SimpleDirectoryReader(
    #     input_files=[
    #         "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/scraped_urls/www_imdb_com_title_tt32812118.md"]
    # )
    # docs = reader.load_data()
    # texts = [doc.text for doc in docs]

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    output_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/job-postings.json"
    data = load_file(data_file)

    json_attributes = [
        # "id",
        "link",
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
        text_chunks = splitter.split_text(text_content)
        data_chunks.append(text_chunks)

    question = 'Given the job posting details provide the data for a job applicant.'
    for text_chunks in tqdm(data_chunks, total=len(data_chunks), unit="chunk"):
        # Summarize
        summarizer = Summarizer(llm=settings_manager.llm)
        response = summarizer.summarize(question, text_chunks)

        job_postings.append(response.__dict__)

        # Inspect response
        logger.success(format_json(response.__dict__))

        save_file(job_postings, output_file)


if __name__ == "__main__":
    main()
