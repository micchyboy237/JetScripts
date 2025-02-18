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
    valid_json_incorrect_sample = "{\"jobTitle\":\"Link Building Specialist\",\"jobType\":\"Full Time\",\"description\":null,\"qualifications\":{\"mandatory\":[\"Ambitious: We’re growing FAST. A thirst for growth and quality of work to match will be compensated.\",\"Impeccable attention to detail: You read through, take notes, remember details like people’s names, etc. (btw my name is Bronwyn,  and if you send the application to any other name or Sir, Sir/Madam, Hiring Manager, etc., I won’t read it ;).\",\"Competent: You are an experienced link builder and have knowledge about SEO.\",\"An A player: You don’t just do “your best”, you do what’s required.\",\"A team player: With a fast-growing team, everyone has to care about the other team members’ success.\",\"A good communicator: You’ll be capturing order details from clients, sometimes with plenty of little nuances / specific requirements. You need to understand clearly what was requested and ensure the message gets across to the fulfillment team clearly.\",\"Honest: You come forward when you mess up and communicate efficiently with your manager so you can fix the issue together. Everybody makes mistakes, it’s how we react to them that makes a difference.\",\"Effective: You get the most out of the hours you work.\"],\"preferred\":[]},\"responsibilities\":null,\"company\":\"Bronwyn Reynolds\",\"industry\":null,\"location\":{\"city\":null,\"state\":null,\"country\":null,\"remote\":true},\"skills\":[\"Fluent spoken and written English absolutely required.\",\"2 years+ previous experience as a link builder or SEO specialist\",\"Proficiency in Excel/Google Sheets is desired\",\"Script (Google Sheets) development skills are highly valued although not necessarily required.\"],\"tools\":null,\"collaboration\":[\"A team player: With a fast-growing team, everyone has to care about the other team members’ success.\"],\"workArrangement\":{\"schedule\":null,\"hoursPerWeek\":null,\"remote\":true},\"compensation\":{\"salaryRange\":{\"min\":550,\"max\":null,\"currency\":\"USD\"},\"benefits\":[\"Full Social Benefits Package after trial period.\",\"13th month Payment (eligible after trial period) paid on December 15th\",\"Paid holidays (national holidays + Christmas to New Year week + 2 weeks per year)\"]},\"applicationProcess\":{\"applicationLinks\":[\"https://www.onlinejobs.ph/jobseekers/job/1198646\"],\"contactInfo\":null,\"instructions\":[\"Send the following documents to bronwyn@getmelinks.com:\",\"- a copy of your resume\",\"- at least two contacts of reference (name, work relationship with you, and contact method; we WILL conduct a background check)\",\"- the full breakdown of your DISC personality as per www.123test.com/disc-personality-test/ (do a new test and send a capture of the final result)\"]},\"postedDate\":\"2025-02-18\"}"
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
    country: Optional[str] = Field(
        None, description="Country where the job is located")
    remote: Optional[bool] = Field(
        None, description="Indicates if remote work is allowed")


class Qualifications(BaseModel):
    mandatory: Optional[List[str]] = Field(
        None, description="Required qualifications, skills, and experience")
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
    currency: Optional[str] = Field(
        None, description="Currency of the salary (e.g., USD, EUR)")


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
    jobTitle: Optional[str] = Field(
        "", description="Title of the job position")
    jobType: Optional[str] = Field(
        "", description="Type of employment (e.g., Full-Time, Part-Time, Contract, Internship)")
    description: Optional[str] = Field("", description="Brief job summary")
    qualifications: Optional[Qualifications] = Field(
        None, description="Job qualifications and requirements")
    responsibilities: Optional[List[str]] = Field(
        None, description="List of job responsibilities")
    company: Optional[str] = Field(
        None, description="Name of the hiring company or employer")
    industry: Optional[str] = Field(
        None, description="Industry related to the job (e.g., Technology, Healthcare, Finance)")
    location: Optional[Location] = Field(
        None, description="Job location details")
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


SampleModel = JobPosting


if __name__ == "__main__":
    # main_valid_dict_correct()
    # main_valid_json_correct()
    main_valid_json_incorrect()
    # main_invalid_json()
