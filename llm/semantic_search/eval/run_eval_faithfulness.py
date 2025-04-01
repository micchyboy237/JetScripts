from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from llama_index.core.evaluation.faithfulness import FaithfulnessEvaluator
from llama_index.core.evaluation.relevancy import RelevancyEvaluator
from llama_index.core.prompts.base import PromptTemplate

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

RELEVANCY_EVAL_TEMPLATE = PromptTemplate("""
Your task is to evaluate whether the response to the query aligns with the provided context information.
You have two options to answer: YES or NO.
Answer YES if the response to the query is in line with the context information; otherwise, answer NO.

Query and Response:
{query_str}

Context:
{context_str}

Answer:
""")

RELEVANCY_REFINE_TEMPLATE = PromptTemplate("""
We want to understand if the following query and response are in line with the context information:
{query_str}
We have provided an existing YES/NO answer:
{existing_answer}
We have the opportunity to refine the existing answer (only if needed) with additional context below.
------------
{context_msg}
------------
If the existing answer was already YES, still answer YES.
If the information is present in the new context, answer YES.
Otherwise, answer NO.
""")

eval_model = "gemma3:4b"

eval_llm = Ollama(model=eval_model)
relevancy_evaluator = FaithfulnessEvaluator(
    llm=eval_llm,
)

# Evaluate context relevancy
relevancy_eval_result = relevancy_evaluator.evaluate(
    query=query,
    response=output,
    contexts=[context],
)

logger.success(format_json(relevancy_eval_result))
copy_to_clipboard(relevancy_eval_result)
