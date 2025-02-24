from jet.vectors.ner import load_nlp_pipeline, extract_entities_from_text
from pydantic import BaseModel
from typing import List


NER_MODEL = "urchade/gliner_small-v2.1"
NER_STYLE = "ent"


# Request Models

class TextRequest(BaseModel):
    text: str


class ProcessRequest(BaseModel):
    model: str = NER_MODEL
    labels: List[str]
    style: str = NER_STYLE
    data: List[TextRequest]
    chunk_size: int = 250


class SingleTextRequest(BaseModel):
    text: str
    model: str = NER_MODEL
    labels: List[str]
    style: str = NER_STYLE
    chunk_size: int = 250


# Response Models

class Entity(BaseModel):
    text: str
    label: str
    score: float


class ProcessedTextResponse(BaseModel):
    text: str
    entities: List[Entity]


class ProcessResponse(BaseModel):
    data: List[ProcessedTextResponse]


def extract_entity(request: SingleTextRequest):
    nlp = load_nlp_pipeline(request.model, request.labels,
                            request.style, request.chunk_size)
    return extract_entities_from_text(nlp, request.text)


def extract_entities(request: ProcessRequest):
    results = []
    nlp = load_nlp_pipeline(request.model, request.labels,
                            request.style, request.chunk_size)

    for item in request.data:
        entities = extract_entities_from_text(nlp, item.text)
        results.append(ProcessedTextResponse(
            text=item.text, entities=entities))

    return ProcessResponse(data=results)


def main():

    sample_text = "Job Title: Join our awesome team! We are in need of a Full-Stack Developer (Full time/Long Term)\n\nTechnical Skillset:\n\n • Has a strong understanding of PHP, CSS and HTML.\n\n • Has experience in building web applications and working with  Javascript frameworks.\n\n • Familiarity with Node.js is a plus, though willingness to self-learn is essential.\n\n • Has experience in creating and consuming REST API endpoints.\n\n • Has strong understanding of API design principles (authentication, validation, error handling).\n\n • Proficiency in backend development with any programming language (Python, Java, Ruby, etc.)\n\n • Has a strong experience with React.\n\n • Has an experience with other modern Javascript frameworks such as Angular, Vue, Svelte or Solid.\n\n • Proficiency in working with PostgreSQL and query optimization.\n\n • Has experience working with N8N or similar workflow automation tools such as Make, Powerautomate or Zapier.\n\n • Has the ability to integrate and automate complex workflows to enhance system functionality.\n\n • Proficiency with cloud systems (AWS, Azure)\n\n • Has a strong understanding of Generative AI and its difference from traditional AI.\n\n • Familiarity with the applications and concepts of generative models in AI systems.\n\n • Has a basic understanding of Retrieval-Augmented Generation (RAG) and its application in AI systems.\n\n • Familiarity with fine tuning of AI models and understanding how it differs from RAG.\n\n • Has basic knowledge of function calling and how it enhances the capabilities of AI Systems.\n\n\n\nMusts:\n\n• Must be working full time\n\n• Must have own laptop, webcam and a fast internet\n\n \n\nPerks:\n\n • Permanent Full-time Work (40hrs/week)\n\n • Flexible Working Hours\n\n • Work from Home Setup\n\n • Paid Overtime Work\n\n • Paid Leaves including Vacation, Sick and Maternal/Paternal Leaves\n\n • HMO\n\n • Follows Philippine Standard Time\n\n • Follows Philippine Holiday Schedule\n\n • Long Term Perspective including Bonuses\n\n • Quarterly Onsite Team Events\n\n\n\nAbout Us:\n\nPerulatus GmBH is a start-up company that focuses on digitally processing care services applications and upgrade for insurance companies in Germany. Our team is looking for a long term support to help us design our website.\n\n\n\nHow to Apply:\n\n1. Change the subject like to “I Want to Work for [Insert the name of the company found on the imprint of this page: \nUpgrade to see actual info\n – Fullstack Developer”\n\n2. Take a screenshot of your internet connection.\n\n3. At the top of your message write 2-3 sentences on why you would like this position, and why you are a good fit.\n\n\n\nDo not write more than that or else your application will be deleted. Make sure you follow the instructions. I will only look for applicants that really want the job. The next step will be a zoom call and video will be required.\n\n\n\nThank you and have a good day.\n\n\n\nQuin"
    labels = ["role", "application", "technology stack", "qualifications"]
    chunk_size = 250

    # Calling extract_entity
    single_request = SingleTextRequest(
        text=sample_text,
        labels=labels,
        chunk_size=chunk_size
    )
    entity_result = extract_entity(single_request)
    print("Extracted Entity:", entity_result)

    # Calling extract_entities
    batch_request = ProcessRequest(
        labels=labels,
        chunk_size=chunk_size,
        data=[TextRequest(text=sample_text), TextRequest(
            text="Google acquired DeepMind.")]
    )
    entities_result = extract_entities(batch_request)
    print("Extracted Entities:", entities_result)


if __name__ == "__main__":
    main()
