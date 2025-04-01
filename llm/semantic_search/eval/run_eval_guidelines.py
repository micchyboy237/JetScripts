import asyncio
import json
import os
from typing import Type
from jet.file.utils import load_file, save_file
from jet.llm.helpers.qa_dataset_generator import QADatasetGenerator
from jet.llm.helpers.question_generator import QuestionGenerator
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from jet.llm.evaluators.guideline_evaluator import GuidelineEvaluator
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.transformers.object import make_serializable
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field

# Sample inputs
topic = "Philippine national ID registration tips 2025"
query = f"Given the context information, extract all data relevant to the topic. Output as a structured JSON object.\nTopic: {topic}"
output = """
<!-- Answer 1 -->

# Philippines National ID Registration Tips 2025

The Philippine Identification System (PhilSys) is a government initiative aimed at simplifying identification processes for all Filipino citizens and resident aliens. This guide will walk you through the registration process, requirements, tracking, and benefits of obtaining a National ID in the Philippines in 2025.

## Key Features of the PhilID
- The PhilID is equipped with essential features such as a unique 12-digit PhilSys Number (PSN), biometric data (fingerprints, iris scan, and photo), and demographic details like name, birthdate, and address.
- It also includes a QR code for secure and seamless digital identity verification.

## How to Register for the National ID
Register Now

1. **Online Pre-Registration**
   - Visit [register.philsys.gov.ph](http://register.philsys.gov.ph).
   - Fill out your demographic details such as name, birthdate, address, and contact information.
   - Save your Application Reference Number (ARN) or QR code after completing the form.
   - Book an appointment at your preferred registration center.

2. **Biometrics Capture**
   - Visit the registration center on your scheduled appointment date.
   - Bring your ARN/QR code and required documents (see requirements below).
   - Provide biometric data (fingerprints, iris scans, and photograph).

3. **Delivery of PhilID**
   - After registration, you will receive a transaction slip with a tracking reference number.
   - The PhilID will be delivered to your registered address by PHLPost.

## Where Can You Register?
- PSA Regional and Provincial Offices
- Local Government Units (LGUs)
- Malls like SM Supermalls
- Mobile units (PhilSys on Wheels or PhilSys on Boat) for remote areas
- For Overseas Filipino Workers (OFWs), registration is available at Philippine embassies or consulates.

## Requirements for National ID Registration
### Primary Documents:
- PSA-issued Birth Certificate
- Passport or ePassport
- Unified Multi-purpose Identification Card (UMID)
- Driver's License

### Secondary Documents (if primary IDs are unavailable):
- School ID
- Barangay Certificate
- Voter's ID
- NBI Clearance

For children under five years old:
- Only demographic information and a front-facing photo are required.
- Their PSN will be linked to their parent or legal guardian's record.

## Tracking Your National ID
If you've completed registration but haven't received your PhilID:
- Visit [PHLPost Tracking](https://www.phlpost.gov.ph/tracking).
- Enter your transaction reference number (TRN) to check delivery status.
- If there's no result, it means your ID is still being processed.

## Benefits of the National ID
- Simplified Transactions: Use one card for banking, government services, and private transactions.
- Improved Access to Services: Easier enrollment in social welfare programs like SSS, GSIS, Pag-IBIG Fund, and PhilHealth.
- Digital Integration: Enables secure identity verification through QR codes.
- Inclusivity: Ensures access for marginalized groups across urban and rural areas.
""".strip()
context = """
# National ID Registration in the Philippines
Bygillani January 21, 2025February 16, 2025
The Philippine Identification System (PhilSys) is a government initiative aimed at simplifying identification processes for all Filipino citizens and resident aliens. This guide will walk you through the registration process, requirements, tracking, and benefits of obtaining a National ID in the Philippines.
Get National ID
Table of Contents
* What is the National ID?
* Key Features of the PhilID
* How to Register for the National ID
* Online Pre-Registration
* Biometrics Capture
* Delivery of PhilID
* Where Can You Register?
* Requirements for National ID Registration
* Primary Documents:
* Secondary Documents (if primary IDs are unavailable):
* Tracking Your National ID
* Benefits of the National ID
* Online vs Walk-In Registration
* Summary Table of Steps
* FAQs
* Is registration for the National ID mandatory?
* How can I track the delivery of my National ID?
* What documents are required for registration?
* Can minors register for the National ID?
* What should I do if there's an error in my PhilID?
* Conclusion
## What is the National ID?
The National ID, also known as the PhilSys ID or PhilID, is a single, government-issued identification card that consolidates multiple forms of identification. It contains demographic and biometric information to ensure secure and accurate identification.
## Key Features of the PhilID
The PhilID is equipped with essential features such as a unique 12-digit PhilSys Number (PSN), biometric data (fingerprints, iris scan, and photo), and demographic details like name, birthdate, and address. It also includes a QR code for secure and seamless digital identity verification.
* PhilSys Number (PSN): A unique 12-digit permanent identification number.
* Biometric Data: Includes fingerprints, iris scans, and a photo.
* Demographic Information: Name, birthdate, address, and blood type.
* QR Code: Allows digital verification for faster transactions.
## How to Register for the National ID
Register Now
Registering for the National ID involves three steps: online pre-registration, biometric data capture at a registration center, and delivery of the PhilID to your address. The process is designed to be simple and accessible for all Filipinos and resident aliens.
## Online Pre-Registration
Online pre-registration allows applicants to fill out their demographic details on the PhilSys website and secure an appointment at their preferred registration center. This step streamlines the process by reducing waiting times during in-person registration.
* Visit register.philsys.gov.ph.
* Fill out your personal details such as name, birthdate, address, and contact information.
* Save your Application Reference Number (ARN) or QR code after completing the form.
* Book an appointment at your preferred registration center.
## Biometrics Capture
Biometrics capture is an in-person process where applicants provide their fingerprints, iris scans, and a photograph at the registration center. This step ensures the uniqueness and security of each PhilID issued.
* Visit the registration center on your scheduled appointment date.
* Bring your ARN/QR code and required documents (see requirements below).
* Provide biometric data (fingerprints, iris scans, and photograph).
## Delivery of PhilID
The PhilID is delivered to the applicant's registered address via PHLPost after successful registration. Applicants can track the delivery status using their transaction reference number (TRN).
* After registration, you will receive a transaction slip with a tracking reference number.
* The PhilID will be delivered to your registered address by PHLPost.
Step| Description| Requirements
---|---|---
1| Online Pre-registration| Personal details
2| Biometrics Capture| ARN/QR code & documents
3| Delivery of PhilID| Transaction slip
### Where Can You Register?
You can register for the National ID at PSA offices, LGUs, mall-based centers like SM Supermalls, or through mobile units such as PhilSys on Wheels. These centers ensure accessibility for Filipinos nationwide, including those in remote areas.
* PSA Regional and Provincial Offices
* Local Government Units (LGUs)
* Malls like SM Supermalls
* Mobile units (PhilSys on Wheels or PhilSys on Boat) for remote areas
For Overseas Filipino Workers (OFWs), registration is available at Philippine embassies or consulates.
### Requirements for National ID Registration
To register for the National ID, applicants need a primary document like a PSA-issued Birth Certificate, Passport, or UMID. Secondary documents such as a school ID or Barangay Certificate can be used if primary IDs are unavailable.
#### Primary Documents:
* PSA-issued Birth Certificate
* Passport or ePassport
* Unified Multi-purpose Identification Card (UMID)
* Driver's License
### Secondary Documents (if primary IDs are unavailable):
* School ID
* Barangay Certificate
* Voter's ID
* NBI Clearance
For children under five years old:
* Only demographic information and a front-facing photo are required.
* Their PSN will be linked to their parent or legal guardian's record.
### Tracking Your National ID
If you've completed registration but haven't received your PhilID:
* Visit PHLPost Tracking.
* Enter your transaction reference number (TRN) to check delivery status.
* If there's no result, it means your ID is still being processed.
Tracking Method| Details
---|---
PHLPost Website| Use TRN from your transaction slip to check delivery status.
PSA Hotline| Call 1388 for assistance with tracking issues.
Authorization Letter| If unavailable during delivery, leave an authorization letter along with a valid ID copy.
### Benefits of the National ID
* Simplified Transactions: Use one card for banking, government services, and private transactions.
* Improved Access to Services: Easier enrollment in social welfare programs like SSS, GSIS, Pag-IBIG Fund, and PhilHealth.
* Digital Integration: Enables secure identity verification through QR codes.
* Inclusivity: Ensures access for marginalized groups across urban and rural areas.
### Online vs Walk-In Registration
Feature| Online Pre-registration| Walk-In Registration
---|---|---
Appointment Booking| Required| Not required
Document Submission| At registration center| At registration center
Biometric Data Capture| In-person| In-person
Processing Time| Faster| May take longer
""".strip()

GUIDELINES = [
    "The response should fully answer the query.",
    "The response should avoid being vague or ambiguous.",
    (
        "The response should be specific and use statistics or numbers when"
        " possible."
    ),
]


class GuidelineEvalResult(EvaluationResult):
    guideline: str


class GuidelineEvalResultSchema(BaseModel):
    passing: bool = Field(
        ..., description="Indicates whether the response meets the evaluation guidelines.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Evaluation score ranging from 0.0 (lowest) to 1.0 (highest), representing guideline adherence. Scores 0.5 and above are considered passing."
    )
    feedback: str = Field(
        ...,
        description="Detailed feedback on the response, highlighting strengths and areas for improvement."
    )

    @property
    def passing(self) -> bool:
        return self.score >= 0.5


def run_evaluate_guidelines(model: OLLAMA_MODEL_NAMES, query: str, contexts: list[str], response: str, guidelines: list[str], output_cls: Type[BaseModel] = GuidelineEvalResultSchema) -> list[GuidelineEvalResult]:
    llm = Ollama(model=model)
    output_parser = PydanticOutputParser(output_cls=output_cls)

    evaluators = [
        GuidelineEvaluator(llm=llm, guidelines=guideline,
                           output_parser=output_parser)
        for guideline in guidelines
    ]

    results = []
    for guideline, evaluator in zip(guidelines, evaluators):
        eval_result = evaluator.evaluate(
            query=query,
            contexts=contexts,
            response=response,
        )
        print("=====")
        print(f"Guideline: {guideline}")
        print(f"Passing: {eval_result.passing}")
        print(f"Score: {eval_result.score}")
        print(f"Feedback: {eval_result.feedback}")
        results.append(GuidelineEvalResult(
            guideline=guideline, **eval_result.model_dump()))
    return results


def main():
    eval_model = "gemma3:4b"
    output_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/semantic_search/eval/generated/run_eval_guidelines/guideline_eval_results.json"

    guideline_results: list[list[GuidelineEvalResult]] = []
    guideline_eval_results = run_evaluate_guidelines(
        model=eval_model,
        query=query,
        response=output,
        contexts=[context],
        guidelines=GUIDELINES
    )
    guideline_results.append(guideline_eval_results)
    logger.newline()
    logger.info(f"Evaluated guideline result:")
    logger.success(format_json(guideline_eval_results))
    save_file(guideline_results, output_file)

    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
