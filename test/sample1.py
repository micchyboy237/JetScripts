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


data_str = """
```json
{
  "data": [
    {
      "question": "What is your most recent work experience, and what was your role?",
      "answer": "My most recent work experience is at JulesAI (formerly Macanta Software Ltd.) as a Web / Mobile Developer. I have been working on developing a white label CRM system for different businesses since Jul 2020."
    },
    {
      "question": "Can you tell me about your experience with React and React Native?",
      "answer": "I have extensive experience with React and React Native, having used them in my previous roles at JulesAI (formerly Macanta Software Ltd.), 8WeekApp, ADEC Innovations, Asia Pacific Digital, and Entertainment Gateway Group (now Yondu)."
    },
    {
      "question": "How do you handle client feedback and requirements?",
      "answer": "In my current role at JulesAI (formerly Macanta Software Ltd.), I maintain and improve the white label CRM system based on client feedback and requirements. This involves working closely with clients to understand their needs and making necessary changes to ensure they are satisfied."
    },
    {
      "question": "What was your experience like working on a social networking app for students, parents, teachers, and schools?",
      "answer": "I worked on the Graduapp project at 8WeekApp as a Web / Mobile Developer. The app served as an online journal of their experience as a student at their institution. I enjoyed working on this project and learned a lot from it."
    ],
    {
      "question": "Can you describe your experience with AWS?",
      "answer": "In my current role at JulesAI (formerly Macanta Software Ltd.), I have been using AWS to host the white label CRM system. I have hands-on experience with setting up and configuring AWS services to meet the needs of our clients."
    },
    {
      "question": "How do you stay updated with new technologies?",
      "answer": "I make it a point to regularly learn about new technologies and frameworks, especially those relevant to my field. This helps me stay ahead in terms of skills and knowledge, which is essential for delivering high-quality work."
    },
    {
      "question": "Can you tell me about your experience with Node.js?",
      "answer": "I have used Node.js in my previous roles at 8WeekApp, ADEC Innovations, and Asia Pacific Digital. I have hands-on experience with setting up and configuring Node.js for various projects."
    },
    {
      "question": "What was your role like at Entertainment Gateway Group (now Yondu)?",
      "answer": "I worked as a Web Developer at Entertainment Gateway Group (now Yondu) from Jun 2012 to Nov 2014. During this time, I worked on features for an insurance web app."
    },
    {
      "question": "Can you describe your experience with AngularJS?",
      "answer": "I used AngularJS in my previous role at Asia Pacific Digital. I have hands-on experience with setting up and configuring AngularJS for various projects."
    }
  ]
}
```
"""

QuestionAnswer.model_validate_json(json.dumps(data_str))
logger.success("Valid format!")
