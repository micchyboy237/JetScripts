import nest_asyncio
from jet.llm.ollama.base import Ollama
from llama_index.core.evaluation import GuidelineEvaluator
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/guideline_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Guideline Evaluator

# This notebook shows how to use `GuidelineEvaluator` to evaluate a question answer system given user specified guidelines.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.


initialize_ollama_settings()


nest_asyncio.apply()

GUIDELINES = [
    "The response should fully answer the query.",
    "The response should avoid being vague or ambiguous.",
    (
        "The response should be specific and use statistics or numbers when"
        " possible."
    ),
]

llm = Ollama(temperature=0, model="llama3.1")

evaluators = [
    GuidelineEvaluator(llm=llm, guidelines=guideline)
    for guideline in GUIDELINES
]

sample_data = {
    "query": "'Can you tell me about your current position and responsibilities at JulesAI (formerly Macanta Software Ltd.)?'",
    "contexts": ['Companies\n\nJob History (from most recent)\n1.) Jul 2020 - Present\n- JulesAI (formerly Macanta Software Ltd.)\n- Position: Web / Mobile Developer\n- Task: Developed a white label CRM system for different businesses that is customizable to align with specific workflows and requirements.\n- Currently maintaining and improving the system based on client feedback and requirements.\n- Key technologies: React, React Native, AWS\n\n2.) Jan 2019 - Jun 2020\n- 8WeekApp\n- Position: Web / Mobile Developer\n- Task: Developed a social networking app (Graduapp) for students, parents, teachers, and schools. The app serves as an online journal of their experience as a student at their institution.\n- Key technologies: React, React Native, Node.js, Firebase, MongoDB\n\n3.) Nov 2016 - Jan 2019\n- ADEC Innovations\n- Position: Web / Mobile Developer\n- Task: Worked on projects ranging from in-house to outsourced web and mobile applications.\n- Key technologies: React, React Native, Node.js, Firebase Firestore, Google Cloud Functions\n\n4.) Nov 2014 - Sep 2016\n- Asia Pacific Digital\n- Position: Web / Mobile App Developer\n- Task: Worked on web and mobile projects.\n- Key technologies: AngularJS, Ionic, jQuery\n\n5.) Jun 2012 - Nov 2014\n- Entertainment Gateway Group (now Yondu)\n- Position: Web Developer\n- Task: Worked on features for an insurance web app.\n- Key technologies: Java, JavaScript, jQuery', 'Web apps\n\na.) Jules Procure\nAchievements:\n- Started as the sole client side developer, built enterprise web and mobile CRM apps starting from provided mockups to production\n- JulesAI CEO was impressed and acquired ownership of existing CRM\n- Successfully integrated existing CRM with JulesAI\'s workflow to be rebranded as "Jules Procure"\n- Key features: Contact dashboard, Data builder, Task calendar, Workflow boards, Form builders, Price list generator, Automated emails based on triggers, and more\n- Technologies used: React, React Native, AWS Lambdas, GraphQL, Docker, Serverless, Jest\n\nb.) Digital Cities PH\nAchievements:\n- As the lead developer, I worked on a portal that showcases the profiles of provinces and cities in the Philippines\n- Developed an interactive Philippine map with clickable provinces, enabling users to access detailed descriptions and statistics for each region\n- Key features: Interactive map, Search, Filtering, Fast loading, SEO-friendly\n- Technologies used: React, GraphQL, React Static, Headless CMS\n\nc.) ADEC Kenya, AMDATEX\nAchievements:\n- Built UI components from mockups using Photoshop to achieve pixel-perfect look\n- Key features: Responsive, Reusable components\n- Technologies used: React, jQuery, Wordpress'],
    "response": "I currently hold the position of Web / Mobile Developer at JulesAI. My primary responsibility is maintaining and improving a white label CRM system that I developed for various businesses, ensuring it meets specific workflows and requirements through client feedback and input. This involves utilizing key technologies such as React, React Native, and AWS to continuously enhance the system's functionality and user experience."
}

for guideline, evaluator in zip(GUIDELINES, evaluators):
    eval_result = evaluator.evaluate(
        query=sample_data["query"],
        contexts=sample_data["contexts"],
        response=sample_data["response"],
    )
    print("=====")
    print(f"Guideline: {guideline}")
    print(f"Pass: {eval_result.passing}")
    print(f"Feedback: {eval_result.feedback}")

logger.info("\n\n[DONE]", bright=True)
