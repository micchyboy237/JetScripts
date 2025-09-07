from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"

model_name = "gpt-4o"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

role = "travel agent"
company = "contoso travel"
responsibility = "booking flights"

response = client.complete(
    messages=[
        SystemMessage(content="""You are an expert at creating AI agent assistants.
You will be provided a company name, role, responsibilities and other
information that you will use to provide a system prompt for.
To create the system prompt, be descriptive as possible and provide a structure that a system using an LLM can better understand the role and responsibilities of the AI assistant."""),
        UserMessage(content=f"You are {role} at {company} that is responsible for {responsibility}."),
    ],
    model=model_name,
    temperature=1.,
    max_tokens=1000,
    top_p=1.
)

logger.debug(response.choices[0].message.content)

logger.info("\n\n[DONE]", bright=True)