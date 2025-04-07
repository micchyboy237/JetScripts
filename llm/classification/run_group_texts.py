from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.llm.ollama.base import Ollama
prompt = """You are an intelligent assistant that analyzes structured or semi-structured text and groups it into meaningful labeled sections.

Given the following text, split it into clearly labeled groups based on content and structure. Each group should have:
- A concise group title
- The corresponding content that belongs to that group

Output format:
[
  {
    "label": "Section Title",
    "content": "Associated content lines from the text..."
  },
  ...
]

Text:
"""


text = """
Key Responsibilities:
Develop and maintain high-quality mobile applications using React Native.
Deploy applications to the Apple App Store and Google Play Store.
Implement efficient navigation and project structuring using Expo Router.
Design and build user interfaces with React Native Paper for consistent and accessible app design.
Style components using NativeWind and Tailwind CSS utilities.
Integrate and manage push notifications with services like Firebase Cloud Messaging and APNs.
Collaborate with backend teams to integrate RESTful APIs into mobile apps.
Utilize Zustand or similar libraries for effective state management.
Handle form inputs and validations with React Hook Form and Zod.
Debug and resolve application issues to ensure optimal performance.
Requirements:
Experience:
Minimum of 3 years of professional experience in mobile application development using React Native.
Proven track record of building and maintaining mobile applications deployed on both iOS and Android platforms.
Technical Skills:
Strong proficiency in React Native and its core principles.
Hands-on experience deploying apps to the Apple App Store and Google Play Store.
Proficiency with Expo Router for navigation and project structuring.
Experience with React Native Paper for consistent app design.
Familiarity with NativeWind for styling components using Tailwind CSS utilities.
Knowledge of push notification services such as Firebase Cloud Messaging and APNs.
Experience working with native modules and integrating them with React Native.
Understanding of RESTful APIs and integration with mobile applications.
Strong debugging skills to troubleshoot and resolve issues effectively.
Proficiency in Zustand or similar state management libraries.
Experience with React Hook Form for efficient form handling.
Knowledge of Zod for schema validation.
Knowledge of TypeScript.
## Employer questions
Your application will include the following questions:
* Which of the following types of qualifications do you have?
* What's your expected monthly basic salary?
* How many years' experience do you have as a React Native Developer?
### Company profile
#### Cafisglobal Inc.
Information & Communication Technology11-50 employees
Cafisglobal Inc is a boutique ITBPO company servicing clients globally. We provide both voice support and software development services to our clients. We are small but growing organization expanding to new markets and is seeking experienced and dedicated workers to join our team.
Cafisglobal Inc is a boutique ITBPO company servicing clients globally. We provide both voice support and software development services to our clients. We are small but growing organization expanding to new markets and is seeking experienced and dedicated workers to join our team.
More about this company
"""

# Use this full_prompt with your preferred LLM (e.g., OpenAI, Anthropic, Mistral, etc.)
full_prompt = prompt + text.strip()

# llm = Ollama(model="mistral")
llm = Ollama(model="llama3.2")
response = llm.chat(full_prompt)

logger.success(format_json(response))
