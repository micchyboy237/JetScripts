import json
from jet.logger import logger
from pydantic.fields import Field
from pydantic.main import BaseModel


class Data(BaseModel):
    question: str = Field(
        description="Short question text answering partial context information provided.")
    answer: str = Field(
        description="The concise answer to the question given the relevant partial context.")


class QuestionAnswer(BaseModel):
    data: list[Data]


data = {
    "data": [
        {
            "question": "What is your name?",
            "answer": "Jethro Reuel A. Estrada"
        },
        {
            "question": "How old are you?",
            "answer": "34"
        },
        {
            "question": "What is your position?",
            "answer": "Full Stack Web / Mobile Developer"
        },
        {
            "question": "Where are you from?",
            "answer": "Philippines"
        },
        {
            "question": "What is your nationality?",
            "answer": "Filipino"
        },
        {
            "question": "What is your education?",
            "answer": "BS Degree in Computer Engineering (2007 - 2012)"
        },
        {
            "question": "Which school did you attend?",
            "answer": "De La Salle University - Manila"
        }
    ]
}

QuestionAnswer.model_validate_json(json.dumps(data))
logger.success("Valid format!")
