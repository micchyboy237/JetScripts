import asyncio
import json
import os
from typing import Optional, Type
from jet.file.utils import load_file, save_file
from jet.llm.helpers.qa_dataset_generator import QADatasetGenerator
from jet.llm.helpers.question_generator import QuestionGenerator
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from jet.llm.evaluators.guideline_evaluator import CONTEXT_EVAL_GUIDELINES, GuidelineContextEvaluator, GuidelineEvaluator
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
# context = """
# # National ID Registration in the Philippines
# The Philippine Identification System (PhilSys) is a government initiative aimed at simplifying identification processes for all Filipino citizens and resident aliens. This guide will walk you through the registration process, requirements, tracking, and benefits of obtaining a National ID in the Philippines.
# """.strip()
context = """
# National ID Registration in the Philippines
""".strip()

GUIDELINES = [
    "The response should fully answer the query.",
    "The response should avoid being vague or ambiguous.",
    "The response should be comprehensive, ensuring all relevant information from the context is included and nothing essential is omitted.",
]


class GuidelineEvalResult(EvaluationResult):
    guideline: str


def run_evaluate_response_guidelines(model: OLLAMA_MODEL_NAMES, query: str, contexts: list[str], response: str, guidelines: list[str], output_cls: Optional[Type[BaseModel]] = None) -> list[GuidelineEvalResult]:
    llm = Ollama(model=model, temperature=0.75)
    output_parser = PydanticOutputParser(
        output_cls=output_cls) if output_cls else None

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


def run_evaluate_context_guidelines(model: OLLAMA_MODEL_NAMES, query: str, contexts: list[str], response: str, guidelines: list[str] = CONTEXT_EVAL_GUIDELINES, output_cls: Optional[Type[BaseModel]] = None) -> list[GuidelineEvalResult]:
    llm = Ollama(model=model, temperature=0.75)
    output_parser = PydanticOutputParser(
        output_cls=output_cls) if output_cls else None

    evaluators = [
        GuidelineContextEvaluator(
            llm=llm,
            guidelines=guideline,
            output_parser=output_parser,
        )
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
    guideline_eval_results = run_evaluate_context_guidelines(
        model=eval_model,
        query=query,
        contexts=[context],
        response=output,
    )
    guideline_results.append(guideline_eval_results)
    logger.newline()
    logger.info(f"Evaluated guideline result:")
    logger.success(format_json(guideline_eval_results))
    save_file(guideline_results, output_file)

    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
