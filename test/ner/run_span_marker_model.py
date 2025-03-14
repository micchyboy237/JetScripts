import transformers
import os
from pydantic import BaseModel
from span_marker import SpanMarkerModel
from span_marker.tokenizer import SpanMarkerTokenizer
from jet.logger import logger
os.environ["TOKENIZERS_PARALLELISM"] = "true"


print("transformers:", transformers.__version__)

# Download from the 🤗 Hub
model = SpanMarkerModel.from_pretrained(
    "tomaarsen/span-marker-mbert-base-multinerd"
    # "tomaarsen/span-marker-roberta-large-ontonotes5"
)
tokenizer: SpanMarkerTokenizer = model.tokenizer
# Resolves error in data collator trying to find missing pad_token_id
tokenizer.pad_token_id = tokenizer.pad_token_type_id

# Input text
text = "Explore Exciting Opportunities at Vault Outsourcing: Your Gateway to Offshoring Excellence:\n\nAre you seeking a great career opportunity with exceptional benefits? Look no further! Vault Outsourcing is not just a company; it's a dynamic force offering a new and exciting career path.\nWe believe our people are our greatest asset and foster a family atmosphere that encourages excellence. Join us in redefining offshoring excellence, where your career is valued, and exciting opportunities await. Discover what we can offer - your gateway to a fulfilling career!\nENJOY THESE EXCITING BENEFITS WHEN YOU JOIN OUR AMAZING TEAM!\n\nYour equipment is on us! Laptop provided\n13th Month Pay\nHMO benefits for you and your dependents\nGroup Life Insurance\nMental Health Program - GET FREE CONSULTATION!\nEligibility to our Employee Referral Program FROM DAY 1. Get above-average Referral Bonus for every successful hire\nLeave credits available in your first month! Up to 20 leave credits per year.\nUnused Credits Convertible to Cash*\nAnnual performance review\nCompany events\nRewards and Recognition\nMonthly engagement activities\nJob Description:\n\nThe primary responsibilities of the software engineer are to design, develop, test and maintain software programs for computer operating systems or applications as well as confer with and assist team members and other developers on problems, improvements and modifications to system software and projects.\nResponsibilities:\n\nBuilding front end components as well as wrangling APIs in the back end and ensure the quality of delivery.\nConduct code reviews and keep up with industry best practices and new technologies, learning & implementing latest trends.\nContribute to the team culture and be part of the team.\nOwn certain features within the app and assist with guiding others around these features.\nEnsure responsiveness, collaboration and ensure proactivity in coming forward with answers and solutions to product managers & vendors.\nUnderstand what quality code looks like and should be able to advise other within the team on how to achieve this level of quality.\nGuide and mentor other engineers within the team and be available to answer questions or queries and provide constructive feedback in a respectful manner.\nQualifications:\n\nRobust industry experience in React, Redux and Typescript\nProficient in GraphQL\nExperienced with React Native\nExperience with deployment pipeline technologies like AWS, bitbucket\nUnit testing using tools such as enzyme, Cypress, Jest and Mocha\nExperience with software platform design, Full SDLC, SOLID principles, Design Patterns, unit testing, Security & Compliance\nExperience with CI/CD pipelines including GIT / Bitbucket code repositories and workflows\nWorked in an agile scrum, continuously shipping environment\nYou have a customer centric mindset and care about user experience\nYou're a conceptual, critical and analytical thinker, who has sound problem solving skills.\nHungry to learn new skills, share the knowledge with your team and expand your horizon.\nYou can communicate your ideas coherently, especially to those not technologically proficient.\nAn individual who is pro-active, has a strong sense of ownership, and is able to work autonomously.\nYou are comfortable having difficult conversations and providing constructive feedback in a respectful manner.\nWilling to work on a\nDayshift or Australian Schedule\n\nWilling to work On-Site\n\n\nCompetency Matrix:\n\nTechnical Skills:\n Proficient with monitoring tools and metrics within their team's domain. Systematically debugs issues within a single service. Applies a security lens to engineering work, actively seeking vulnerabilities in code and peer reviews.\nFeedback, Communication, Collaboration:\n Delivers both praise and constructive feedback effectively to team members, peers, managers, and business stakeholders. Communicates effectively, clearly, and concisely in both technical and non-technical subjects, considering the audience.\nDelivery:\n Assists teammates in overcoming obstacles, resolving blockers, and completing tasks. Effectively manages risk, change, and uncertainty within their personal scope of work.\nLeadership:\n Strives for objectivity and self-reflection on biases when making decisions. Mentors teammates in an open, respectful, flexible, and empathetic manner.\nStrategy Impact:\n Has a thorough understanding of their team's domain, and how it contributes to overall business strategy.\n\n## Employer questions\nYour application will include the following questions:\n* What's your expected monthly basic salary?\n* How many years' experience do you have as a Front End Engineer?\n* Do you have experience working within a scrum agile team?\n* Which of the following front end development libraries and frameworks are you proficient in?\n* Do you have experience with responsive / mobile first web development?\n* How many years' experience do you have as a Front End React Developer?\n* How many years' experience do you have as an API Developer?\n* How many years' experience do you have as a React Native Developer?"

logger.info(text)
logger.debug("Predicting...")
# Predict entities
entities = model.predict(text)


class SpanMarkerWord(BaseModel):
    text: str
    lemma: str
    start_idx: int
    end_idx: int
    score: float  # Normalized score

    def __str__(self) -> str:
        """Return a readable string representation of the word."""
        return self.text


results = [
    SpanMarkerWord(
        text=entity['span'],
        lemma=entity['label'],
        score=entity['score'],
        start_idx=entity['char_start_index'],
        end_idx=entity['char_start_index'],
    )
    for entity in entities
]


logger.debug("Extracted Entities:")
for result in results:
    logger.newline()
    logger.log("Text:", result.text, colors=["WHITE", "INFO"])
    logger.log("Lemma:", result.lemma, colors=["WHITE", "SUCCESS"])
    logger.log("Score:", f"{result.score:.4f}", colors=[
               "WHITE", "SUCCESS"])
    logger.log("Start:", f"{result.start_idx}", colors=[
               "WHITE", "SUCCESS"])
    logger.log("End:", f"{result.end_idx}",
               colors=["WHITE", "SUCCESS"])
    logger.log("---")
