from jet.logger import logger
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langgraph.prebuilt import create_react_agent
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
# OpenWeatherMap

This notebook goes over how to use the `OpenWeatherMap` component to fetch weather information.

First, you need to sign up for an `OpenWeatherMap API` key:

1. Go to OpenWeatherMap and sign up for an API key [here](https://openweathermap.org/api/)
2. pip install pyowm

Then we will need to set some environment variables:
1. Save your API KEY into OPENWEATHERMAP_API_KEY env variable

## Use the wrapper
"""
logger.info("# OpenWeatherMap")

# %pip install --upgrade --quiet pyowm



os.environ["OPENWEATHERMAP_API_KEY"] = ""

weather = OpenWeatherMapAPIWrapper()

weather_data = weather.run("London,GB")
logger.debug(weather_data)

"""
## Use the tool
"""
logger.info("## Use the tool")



# os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENWEATHERMAP_API_KEY"] = ""

tools = [weather.run]
agent = create_react_agent("ollama:gpt-4.1-mini", tools)

input_message = {
    "role": "user",
    "content": "What's the weather like in London?",
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

logger.info("\n\n[DONE]", bright=True)