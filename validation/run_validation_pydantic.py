import json
from typing import Optional
from pydantic import BaseModel
from jet.validation import validate_json_pydantic
from jet.transformers import make_serializable
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
    valid_json_incorrect_sample = '{\n  "jobTitle": "UI / UX Designer",\n  "jobType": "Part Time",\n  "description": "We are seeking a part-time UI/UX Designer to join our team for an exciting mobile app project.",\n  "qualifications": {\n    "mandatory": [\n      "Proven experience in UI/UX design, with a strong portfolio showcasing mobile app projects",\n      "Proficiency in Figma, Adobe XD, Sketch, or similar design tools",\n      "Understanding of user-centered design principles and usability best practices"\n    ],\n    "preferred": [\n      "Experience in designing for iOS and Android platforms",\n      "Basic knowledge of front-end development (HTML, CSS, React Native, etc.)",\n      "Experience with motion design or animations in UI"\n    ]\n  },\n  "responsibilities": [\n    "Design and refine user flows, wireframes, and high-fidelity UI for a mobile app",\n    "Create interactive prototypes to demonstrate user experience concepts",\n    "Conduct user research and usability testing to improve design decisions",\n    "Collaborate with developers to ensure design feasibility and smooth implementation"\n  ],\n  "company": "Mika Gan",\n  "industry": null,\n  "location": {\n    "country": "Philippines",\n    "remote": true\n  },\n  "skills": [\n    "UI/UX Design",\n    "Figma",\n    "Adobe XD",\n    "Sketch"\n  ],\n  "tools": [\n    "Figma",\n    "Adobe XD",\n    "Sketch"\n  ],\n  "collaboration": null,\n  "workArrangement": {\n    "schedule": "Flexible",\n    "hoursPerWeek": 10\n  },\n  "compensation": {\n    "salaryRange": {\n      "min": "$10/hr",\n      "max": null,\n      "currency": "USD"\n    },\n    "benefits": []\n  },\n  "applicationProcess": {\n    "contactInfo": [\n      "onlinejobs.ph"\n    ],\n    "instructions": [\n      "Apply through onlinejobs.ph"\n    ]\n  },\n  "postedDate": "2025-02-18"\n}'
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


SampleModel = JobPosting


if __name__ == "__main__":
    # main_valid_dict_correct()
    # main_valid_json_correct()
    main_valid_json_incorrect()
    # main_invalid_json()
