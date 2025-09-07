from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict
import os
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# History and Data Analysis Collaboration System

## Overview
This notebook implements a multi-agent collaboration system that combines historical research with data analysis to answer complex historical questions. It leverages the power of large language models to simulate specialized agents working together to provide comprehensive answers.

## Motivation
Historical analysis often requires both deep contextual understanding and quantitative data interpretation. By creating a system that combines these two aspects, we aim to provide more robust and insightful answers to complex historical questions. This approach mimics real-world collaboration between historians and data analysts, potentially leading to more nuanced and data-driven historical insights.

## Key Components
1. **Agent Class**: A base class for creating specialized AI agents.
2. **HistoryResearchAgent**: Specialized in historical context and trends.
3. **DataAnalysisAgent**: Focused on interpreting numerical data and statistics.
4. **HistoryDataCollaborationSystem**: Orchestrates the collaboration between agents.

## Method Details
The collaboration system follows these steps:
1. **Historical Context**: The History Agent provides relevant historical background.
2. **Data Needs Identification**: The Data Agent determines what quantitative information is needed.
3. **Historical Data Provision**: The History Agent supplies relevant historical data.
4. **Data Analysis**: The Data Agent interprets the provided historical data.
5. **Final Synthesis**: The History Agent combines all insights into a comprehensive answer.

This iterative process allows for a back-and-forth between historical context and data analysis, mimicking real-world collaborative research.

## Conclusion
The History and Data Analysis Collaboration System demonstrates the potential of multi-agent AI systems in tackling complex, interdisciplinary questions. By combining the strengths of historical research and data analysis, it offers a novel approach to understanding historical trends and events. This system could be valuable for researchers, educators, and anyone interested in gaining deeper insights into historical topics.

Future improvements could include adding more specialized agents, incorporating external data sources, and refining the collaboration process for even more nuanced analyses.

### Import required libraries
"""
logger.info("# History and Data Analysis Collaboration System")



load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Initialize the language model
"""
logger.info("### Initialize the language model")

llm = ChatOllama(model="llama3.2")

"""
### Define the base Agent class
"""
logger.info("### Define the base Agent class")

class Agent:
    def __init__(self, name: str, role: str, skills: List[str]):
        self.name = name
        self.role = role
        self.skills = skills
        self.llm = llm

    def process(self, task: str, context: List[Dict] = None) -> str:
        messages = [
            SystemMessage(content=f"You are {self.name}, a {self.role}. Your skills include: {', '.join(self.skills)}. Respond to the task based on your role and skills.")
        ]

        if context:
            for msg in context:
                if msg['role'] == 'human':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'ai':
                    messages.append(AIMessage(content=msg['content']))

        messages.append(HumanMessage(content=task))
        response = self.llm.invoke(messages)
        return response.content

"""
### Define specialized agents: HistoryResearchAgent and DataAnalysisAgent
"""
logger.info("### Define specialized agents: HistoryResearchAgent and DataAnalysisAgent")

class HistoryResearchAgent(Agent):
    def __init__(self):
        super().__init__("Clio", "History Research Specialist", ["deep knowledge of historical events", "understanding of historical contexts", "identifying historical trends"])

class DataAnalysisAgent(Agent):
    def __init__(self):
        super().__init__("Data", "Data Analysis Expert", ["interpreting numerical data", "statistical analysis", "data visualization description"])

"""
### Define the different functions for the collaboration system

#### Research Historical Context
"""
logger.info("### Define the different functions for the collaboration system")

def research_historical_context(history_agent, task: str, context: list) -> list:
    logger.debug("🏛️ History Agent: Researching historical context...")
    history_task = f"Provide relevant historical context and information for the following task: {task}"
    history_result = history_agent.process(history_task)
    context.append({"role": "ai", "content": f"History Agent: {history_result}"})
    logger.debug(f"📜 Historical context provided: {history_result[:100]}...\n")
    return context

"""
#### Identify Data Needs
"""
logger.info("#### Identify Data Needs")

def identify_data_needs(data_agent, task: str, context: list) -> list:
    logger.debug("📊 Data Agent: Identifying data needs based on historical context...")
    historical_context = context[-1]["content"]
    data_need_task = f"Based on the historical context, what specific data or statistical information would be helpful to answer the original question? Historical context: {historical_context}"
    data_need_result = data_agent.process(data_need_task, context)
    context.append({"role": "ai", "content": f"Data Agent: {data_need_result}"})
    logger.debug(f"🔍 Data needs identified: {data_need_result[:100]}...\n")
    return context

"""
#### Provide Historical Data
"""
logger.info("#### Provide Historical Data")

def provide_historical_data(history_agent, task: str, context: list) -> list:
    logger.debug("🏛️ History Agent: Providing relevant historical data...")
    data_needs = context[-1]["content"]
    data_provision_task = f"Based on the data needs identified, provide relevant historical data or statistics. Data needs: {data_needs}"
    data_provision_result = history_agent.process(data_provision_task, context)
    context.append({"role": "ai", "content": f"History Agent: {data_provision_result}"})
    logger.debug(f"📊 Historical data provided: {data_provision_result[:100]}...\n")
    return context

"""
#### Analyze Data
"""
logger.info("#### Analyze Data")

def analyze_data(data_agent, task: str, context: list) -> list:
    logger.debug("📈 Data Agent: Analyzing historical data...")
    historical_data = context[-1]["content"]
    analysis_task = f"Analyze the historical data provided and describe any trends or insights relevant to the original task. Historical data: {historical_data}"
    analysis_result = data_agent.process(analysis_task, context)
    context.append({"role": "ai", "content": f"Data Agent: {analysis_result}"})
    logger.debug(f"💡 Data analysis results: {analysis_result[:100]}...\n")
    return context

"""
#### Synthesize Final Answer
"""
logger.info("#### Synthesize Final Answer")

def synthesize_final_answer(history_agent, task: str, context: list) -> str:
    logger.debug("🏛️ History Agent: Synthesizing final answer...")
    synthesis_task = "Based on all the historical context, data, and analysis, provide a comprehensive answer to the original task."
    final_result = history_agent.process(synthesis_task, context)
    return final_result

"""
### HistoryDataCollaborationSystem Class
"""
logger.info("### HistoryDataCollaborationSystem Class")

class HistoryDataCollaborationSystem:
    def __init__(self):
        self.history_agent = Agent("Clio", "History Research Specialist", ["deep knowledge of historical events", "understanding of historical contexts", "identifying historical trends"])
        self.data_agent = Agent("Data", "Data Analysis Expert", ["interpreting numerical data", "statistical analysis", "data visualization description"])

    def solve(self, task: str, timeout: int = 300) -> str:
        logger.debug(f"\n👥 Starting collaboration to solve: {task}\n")

        start_time = time.time()
        context = []

        steps = [
            (research_historical_context, self.history_agent),
            (identify_data_needs, self.data_agent),
            (provide_historical_data, self.history_agent),
            (analyze_data, self.data_agent),
            (synthesize_final_answer, self.history_agent)
        ]

        for step_func, agent in steps:
            if time.time() - start_time > timeout:
                return "Operation timed out. The process took too long to complete."
            try:
                result = step_func(agent, task, context)
                if isinstance(result, str):
                    return result  # This is the final answer
                context = result
            except Exception as e:
                return f"Error during collaboration: {str(e)}"

        logger.debug("\n✅ Collaboration complete. Final answer synthesized.\n")
        return context[-1]["content"]

"""
### Example usage
"""
logger.info("### Example usage")

collaboration_system = HistoryDataCollaborationSystem()

question = "How did urbanization rates in Europe compare to those in North America during the Industrial Revolution, and what were the main factors influencing these trends?"

result = collaboration_system.solve(question)

logger.debug(result)

logger.info("\n\n[DONE]", bright=True)