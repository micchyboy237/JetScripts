from IPython import display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.agent_toolkits import AzureAiServicesToolkit
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Azure AI Services Toolkit

This toolkit is used to interact with the `Azure AI Services API` to achieve some multimodal capabilities.

Currently There are five tools bundled in this toolkit:
- **AzureAiServicesImageAnalysisTool**: used to extract caption, objects, tags, and text from images.
- **AzureAiServicesDocumentIntelligenceTool**: used to extract text, tables, and key-value pairs from documents.
- **AzureAiServicesSpeechToTextTool**: used to transcribe speech to text.
- **AzureAiServicesTextToSpeechTool**: used to synthesize text to speech.
- **AzureAiServicesTextAnalyticsForHealthTool**: used to extract healthcare entities.

First, you need to set up an Azure account and create an AI Services resource. You can follow the instructions [here](https://learn.microsoft.com/en-us/azure/ai-services/multi-service-resource) to create a resource. 

Then, you need to get the endpoint, key and region of your resource, and set them as environment variables. You can find them in the "Keys and Endpoint" page of your resource.
"""
logger.info("# Azure AI Services Toolkit")

# %pip install --upgrade --quiet  azure-ai-formrecognizer > /dev/null
# %pip install --upgrade --quiet  azure-cognitiveservices-speech > /dev/null
# %pip install --upgrade --quiet  azure-ai-textanalytics > /dev/null
# %pip install --upgrade --quiet  azure-ai-vision-imageanalysis > /dev/null
# %pip install -qU langchain-community


# os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["AZURE_AI_SERVICES_KEY"] = ""
os.environ["AZURE_AI_SERVICES_ENDPOINT"] = ""
os.environ["AZURE_AI_SERVICES_REGION"] = ""

"""
## Create the Toolkit
"""
logger.info("## Create the Toolkit")


toolkit = AzureAiServicesToolkit()

[tool.name for tool in toolkit.get_tools()]

"""
## Use within an Agent
"""
logger.info("## Use within an Agent")


llm = ChatOllama(temperature=0)
tools = toolkit.get_tools()
prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

agent_executor.invoke(
    {
        "input": "What can I make with these ingredients? "
        + "https://images.ollama.com/blob/9ad5a2ab-041f-475f-ad6a-b51899c50182/ingredients.png"
    }
)

tts_result = agent_executor.invoke(
    {"input": "Tell me a joke and read it out for me."})
audio_file = tts_result.get("output")


audio = display.Audio(data=audio_file, autoplay=True, rate=22050)
display.display(audio)

sample_input = """
The patient is a 54-year-old gentleman with a history of progressive angina over the past several months.
The patient had a cardiac catheterization in July of this year revealing total occlusion of the RCA and 50% left main disease ,
with a strong family history of coronary artery disease with a brother dying at the age of 52 from a myocardial infarction and
another brother who is status post coronary artery bypass grafting. The patient had a stress echocardiogram done on July , 2001 ,
which showed no wall motion abnormalities , but this was a difficult study due to body habitus. The patient went for six minutes with
minimal ST depressions in the anterior lateral leads , thought due to fatigue and wrist pain , his anginal equivalent. Due to the patient's
increased symptoms and family history and history left main disease with total occasional of his RCA was referred for revascularization with open heart surgery.

List all the diagnoses.
"""

agent_executor.invoke({"input": sample_input})

logger.info("\n\n[DONE]", bright=True)
