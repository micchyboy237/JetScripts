from pydantic import BaseModel, Field
import json
from typing import Optional
from pydantic import BaseModel
from jet.validation import validate_json_pydantic
from jet.transformers.object import make_serializable
from jet.logger import logger

from typing import List, Optional
from pydantic import BaseModel, EmailStr, HttpUrl, Field
from datetime import date
from decimal import Decimal


def main_invalid_json():
    logger.newline()
    logger.info("main_invalid_json()...")
    invalid_json_sample = """{
        "name: "John Doe",
        "age": 30
        "email": "johndoe@example.com"
    }"""
    result = validate_json_pydantic(invalid_json_sample, SampleModel)
    logger.success(json.dumps(make_serializable(result), indent=2))


def main_valid_json_incorrect():
    logger.newline()
    logger.info("main_valid_json_incorrect()...")
    # valid_json_incorrect_sample = """{
    #     "name": 1000,
    #     "age": "30",
    #     "email": "johndoe@example.com"
    # }"""
    valid_json_incorrect_sample = '{\n  "jobTitle": "Link Builder or SEO Specialist",\n  "jobType": "",\n  "description": "We are looking for a Link Builder or SEO Specialist to join our team.",\n  "qualifications": {\n    "mandatory": [\n      "Fluent spoken and written English absolutely required.",\n      "2 years+ previous experience as a link builder or SEO specialist",\n      "Proficiency in Excel/Google Sheets is desired"\n    ],\n    "preferred": [\n      "Script (Google Sheets) development skills are highly valued although not necessarily required."\n    ]\n  },\n  "responsibilities": null,\n  "company": "Get Me Links",\n  "industry": "",\n  "location": {\n    "city": "",\n    "state": "",\n    "country": "",\n    "remote": true\n  },\n  "skills": [\n    "Link building",\n    "SEO"\n  ],\n  "tools": [],\n  "collaboration": null,\n  "workArrangement": {\n    "schedule": "Flexible",\n    "hoursPerWeek": null,\n    "remote": true\n  },\n  "compensation": {\n    "salaryRange": {\n      "min": 550,\n      "max": null,\n      "currency": "USD"\n    },\n    "benefits": [\n      "Full Social Benefits Package after trial period.",\n      "13th month Payment (eligible after trial period) paid on December 15th",\n      "Paid holidays (national holidays + Christmas to New Year week + 2 weeks per year)"\n    ]\n  },\n  "applicationProcess": null,\n  "postedDate": null\n}'
    result = validate_json_pydantic(valid_json_incorrect_sample, SampleModel)
    logger.success(json.dumps(make_serializable(result), indent=2))


def main_valid_json_correct():
    logger.newline()
    logger.info("main_valid_json_correct()...")
    valid_json_correct_sample = """{
        "name": "John Doe",
        "age": 30
    }"""
    result = validate_json_pydantic(valid_json_correct_sample, SampleModel)
    logger.success(json.dumps(make_serializable(result), indent=2))


def main_valid_dict_correct():
    logger.newline()
    logger.info("main_valid_json_correct()...")
    valid_dict_correct_sample = {
        "name": "John Doe",
        "age": 30,
        "email": "johndoe@example.com"
    }
    result = validate_json_pydantic(valid_dict_correct_sample, SampleModel)
    logger.success(json.dumps(make_serializable(result), indent=2))


class Location(BaseModel):
    city: Optional[str] = Field(
        ..., description="City where the job is located")
    state: Optional[str] = Field(
        ..., description="State where the job is located")
    country: Optional[str] = Field(
        ..., description="Country where the job is located")
    remote: Optional[bool] = Field(
        ..., description="Indicates if remote work is allowed")


class Qualifications(BaseModel):
    mandatory: Optional[List[str]] = Field(
        ..., description="Required qualifications, skills, and experience")
    preferred: Optional[List[str]] = Field(
        ..., description="Preferred but not mandatory qualifications")


class TechStack(BaseModel):
    mandatory: Optional[List[str]] = Field(
        ..., description="Mandatory tools, software, or platforms")
    preferred: Optional[List[str]] = Field(
        ..., description="Preferred but not mandatory tools, software, or platforms")


class WorkArrangement(BaseModel):
    schedule: Optional[str] = Field(
        ..., description="Work schedule (e.g., Flexible, Fixed, Shift-based)")
    hoursPerWeek: Optional[int] = Field(
        ..., description="Number of work hours per week")
    remote: Optional[bool] = Field(
        ..., description="Indicates if remote work is allowed")


class SalaryRange(BaseModel):
    min: Optional[int] = Field(..., description="Minimum salary")
    max: Optional[int] = Field(..., description="Maximum salary")
    currency: Optional[str] = Field(
        ..., description="Currency of the salary (e.g., USD, EUR)")


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
    instructions: Optional[List[str]] = Field(
        ..., description="List of instructions on how to apply")


class JobPosting(BaseModel):
    jobTitle: Optional[str] = Field(
        ..., description="Title of the job position")
    jobType: Optional[str] = Field(
        ..., description="Type of employment (e.g., Full-Time, Part-Time, Contract, Internship)")
    description: Optional[str] = Field(..., description="Brief job summary")
    qualifications: Optional[Qualifications] = Field(
        ..., description="Job qualifications and requirements")
    responsibilities: Optional[List[str]] = Field(
        ..., description="List of job responsibilities")
    company: Optional[str] = Field(
        ..., description="Name of the hiring company or employer")
    industry: Optional[str] = Field(
        ..., description="Industry related to the job (e.g., Technology, Healthcare, Finance)")
    location: Optional[Location] = Field(
        ..., description="Job location details")
    techStack: Optional[TechStack] = Field(
        ..., description="Job technological stack requirements")
    collaboration: Optional[List[str]] = Field(
        ..., description="Teams or individuals the candidate will work with")
    workArrangement: Optional[WorkArrangement] = Field(
        ..., description="Work arrangement details")
    compensation: Optional[Compensation] = Field(
        ..., description="Compensation details")
    applicationProcess: Optional[ApplicationProcess] = Field(
        ..., description="Details about how to apply")
    postedDate: Optional[str] = Field(
        ..., description="Date when the job was posted")


SampleModel = JobPosting


if __name__ == "__main__":
    # main_valid_dict_correct()
    # main_valid_json_correct()
    main_valid_json_incorrect()
    # main_invalid_json()
