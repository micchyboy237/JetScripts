from llama_index.agent.openai import OllamaAgent
from llama_index.core.query_engine import SubQuestionQueryEngine
from jet.llm.ollama.base import Ollama
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import load_index_from_storage
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from pathlib import Path
from llama_index.readers.file import UnstructuredReader
import nest_asyncio
import openai
import os
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()


# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]


nest_asyncio.apply()

# NOTE: the code examples assume you're operating within a Jupyter notebook.
# download files
# !mkdir data
# !wget "https://www.dropbox.com/s/948jr9cfs7fgj99/UBER.zip?dl=1" -O data/UBER.zip
# !unzip data/UBER.zip -d data

# !pip install llama-hub unstructured


years = [2022, 2021, 2020, 2019]

loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(
        file=Path(f"./data/UBER/UBER_{year}.html"), split_documents=False
    )
    for d in year_docs:
        d.metadata = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)


Settings.chunk_size = 512
index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(
        doc_set[year],
        storage_context=storage_context,
    )
    index_set[year] = cur_index
    storage_context.persist(persist_dir=f"./storage/{year}")


index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults(
        persist_dir=f"./storage/{year}"
    )
    cur_index = load_index_from_storage(
        storage_context,
    )
    index_set[year] = cur_index


individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[year].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{year}",
            description=f"useful for when you want to answer queries about the {
                year} SEC 10-K for Uber",
        ),
    )
    for year in years
]


query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    llm=Ollama(model="llama3.2", request_timeout=300.0, context_window=4096),
)

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber",
    ),
)

tools = individual_query_engine_tools + [query_engine_tool]


agent = OllamaAgent.from_tools(tools, verbose=True)

response = agent.chat("hi, i am bob")
print(str(response))

# Hello Bob! How can I assist you today?

response = agent.chat(
    "What were some of the biggest risk factors in 2020 for Uber?"
)
print(str(response))

# === Calling Function ===
# Calling function: vector_index_2020 with args: {
#   "input": "biggest risk factors"
# }
# Got output: The biggest risk factors mentioned in the context are:
# 1. The adverse impact of the COVID-19 pandemic and actions taken to mitigate it on the business.
# 2. The potential reclassification of drivers as employees, workers, or quasi-employees instead of independent contractors.
# 3. Intense competition in the mobility, delivery, and logistics industries, with low-cost alternatives and well-capitalized competitors.
# 4. The need to lower fares or service fees and offer driver incentives and consumer discounts to remain competitive.
# 5. Significant losses incurred and the uncertainty of achieving profitability.
# 6. The risk of not attracting or maintaining a critical mass of platform users.
# 7. Operational, compliance, and cultural challenges related to the workplace culture and forward-leaning approach.
# 8. The potential negative impact of international investments and the challenges of conducting business in foreign countries.
# 9. Risks associated with operational and compliance challenges, localization, laws and regulations, competition, social acceptance, technological compatibility, improper business practices, liability uncertainty, managing international operations, currency fluctuations, cash transactions, tax consequences, and payment fraud.
# ========================
# Some of the biggest risk factors for Uber in 2020 were:
#
# 1. The adverse impact of the COVID-19 pandemic and actions taken to mitigate it on the business.
# 2. The potential reclassification of drivers as employees, workers, or quasi-employees instead of independent contractors.
# 3. Intense competition in the mobility, delivery, and logistics industries, with low-cost alternatives and well-capitalized competitors.
# 4. The need to lower fares or service fees and offer driver incentives and consumer discounts to remain competitive.
# 5. Significant losses incurred and the uncertainty of achieving profitability.
# 6. The risk of not attracting or maintaining a critical mass of platform users.
# 7. Operational, compliance, and cultural challenges related to the workplace culture and forward-leaning approach.
# 8. The potential negative impact of international investments and the challenges of conducting business in foreign countries.
# 9. Risks associated with operational and compliance challenges, localization, laws and regulations, competition, social acceptance, technological compatibility, improper business practices, liability uncertainty, managing international operations, currency fluctuations, cash transactions, tax consequences, and payment fraud.
#
# These risk factors highlight the challenges and uncertainties that Uber faced in 2020.

cross_query_str = "Compare/contrast the risk factors described in the Uber 10-K across years. Give answer in bullet points."

response = agent.chat(cross_query_str)
print(str(response))

# === Calling Function ===
# Calling function: sub_question_query_engine with args: {
#   "input": "Compare/contrast the risk factors described in the Uber 10-K across years"
# }
# Generated 4 sub questions.
# [vector_index_2022] Q: What are the risk factors described in the 2022 SEC 10-K for Uber?
# [vector_index_2021] Q: What are the risk factors described in the 2021 SEC 10-K for Uber?
# [vector_index_2020] Q: What are the risk factors described in the 2020 SEC 10-K for Uber?
# [vector_index_2019] Q: What are the risk factors described in the 2019 SEC 10-K for Uber?
# [vector_index_2021] A: The risk factors described in the 2021 SEC 10-K for Uber include the adverse impact of the COVID-19 pandemic on their business, the potential reclassification of drivers as employees instead of independent contractors, intense competition in the mobility, delivery, and logistics industries, the need to lower fares and offer incentives to remain competitive, significant losses incurred by the company, the importance of attracting and maintaining a critical mass of platform users, and the ongoing legal challenges regarding driver classification.
# [vector_index_2020] A: The risk factors described in the 2020 SEC 10-K for Uber include the adverse impact of the COVID-19 pandemic on their business, the potential reclassification of drivers as employees instead of independent contractors, intense competition in the mobility, delivery, and logistics industries, the need to lower fares and offer incentives to remain competitive, significant losses and the uncertainty of achieving profitability, the importance of attracting and retaining a critical mass of drivers and users, and the challenges associated with their workplace culture and operational compliance.
# [vector_index_2022] A: The risk factors described in the 2022 SEC 10-K for Uber include the potential adverse effect on their business if drivers were classified as employees instead of independent contractors, the highly competitive nature of the mobility, delivery, and logistics industries, the need to lower fares or service fees to remain competitive in certain markets, the company's history of significant losses and the expectation of increased operating expenses in the future, and the potential impact on their platform if they are unable to attract or maintain a critical mass of drivers, consumers, merchants, shippers, and carriers.
# [vector_index_2019] A: The risk factors described in the 2019 SEC 10-K for Uber include the loss of their license to operate in London, the complexity of their business and operating model due to regulatory uncertainties, the potential for additional regulations for their other products in the Other Bets segment, the evolving laws and regulations regarding the development and deployment of autonomous vehicles, and the increasing number of data protection and privacy laws around the world. Additionally, there are legal proceedings, litigation, claims, and government investigations that Uber is involved in, which could impose a burden on management and employees and come with defense costs or unfavorable rulings.
# Got output: The risk factors described in the Uber 10-K reports across the years include the potential reclassification of drivers as employees instead of independent contractors, intense competition in the mobility, delivery, and logistics industries, the need to lower fares and offer incentives to remain competitive, significant losses incurred by the company, the importance of attracting and maintaining a critical mass of platform users, and the ongoing legal challenges regarding driver classification. Additionally, there are specific risk factors mentioned in each year's report, such as the adverse impact of the COVID-19 pandemic in 2020 and 2021, the loss of their license to operate in London in 2019, and the evolving laws and regulations regarding autonomous vehicles in 2019. Overall, while there are some similarities in the risk factors mentioned, there are also specific factors that vary across the years.
# ========================
# === Calling Function ===
# Calling function: vector_index_2022 with args: {
#   "input": "risk factors"
# }
# Got output: Some of the risk factors mentioned in the context include the potential adverse effect on the business if drivers were classified as employees instead of independent contractors, the highly competitive nature of the mobility, delivery, and logistics industries, the need to lower fares or service fees to remain competitive, the company's history of significant losses and the expectation of increased operating expenses, the impact of future pandemics or disease outbreaks on the business and financial results, and the potential harm to the business due to economic conditions and their effect on discretionary consumer spending.
# ========================
# === Calling Function ===
# Calling function: vector_index_2021 with args: {
#   "input": "risk factors"
# }
# Got output: The COVID-19 pandemic and the impact of actions to mitigate the pandemic have adversely affected and may continue to adversely affect parts of our business. Our business would be adversely affected if Drivers were classified as employees, workers or quasi-employees instead of independent contractors. The mobility, delivery, and logistics industries are highly competitive, with well-established and low-cost alternatives that have been available for decades, low barriers to entry, low switching costs, and well-capitalized competitors in nearly every major geographic region. To remain competitive in certain markets, we have in the past lowered, and may continue to lower, fares or service fees, and we have in the past offered, and may continue to offer, significant Driver incentives and consumer discounts and promotions. We have incurred significant losses since inception, including in the United States and other major markets. We expect our operating expenses to increase significantly in the foreseeable future, and we may not achieve or maintain profitability. If we are unable to attract or maintain a critical mass of Drivers, consumers, merchants, shippers, and carriers, whether as a result of competition or other factors, our platform will become less appealing to platform users.
# ========================
# === Calling Function ===
# Calling function: vector_index_2020 with args: {
#   "input": "risk factors"
# }
# Got output: The risk factors mentioned in the context include the adverse impact of the COVID-19 pandemic on the business, the potential reclassification of drivers as employees, the highly competitive nature of the mobility, delivery, and logistics industries, the need to lower fares or service fees to remain competitive, the company's history of significant losses and potential future expenses, the importance of attracting and maintaining a critical mass of platform users, and the operational and cultural challenges faced by the company.
# ========================
# === Calling Function ===
# Calling function: vector_index_2019 with args: {
#   "input": "risk factors"
# }
# Got output: The risk factors mentioned in the context include competition with local companies, differing levels of social acceptance, technological compatibility issues, exposure to improper business practices, legal uncertainty, difficulties in managing international operations, fluctuations in currency exchange rates, regulations governing local currencies, tax consequences, financial accounting burdens, difficulties in implementing financial systems, import and export restrictions, political and economic instability, public health concerns, reduced protection for intellectual property rights, limited influence over minority-owned affiliates, and regulatory complexities. These risk factors could adversely affect the international operations, business, financial condition, and operating results of the company.
# ========================
# Here is a comparison of the risk factors described in the Uber 10-K reports across years:
#
# 2022 Risk Factors:
# - Potential adverse effect if drivers were classified as employees instead of independent contractors.
# - Highly competitive nature of the mobility, delivery, and logistics industries.
# - Need to lower fares or service fees to remain competitive.
# - History of significant losses and expectation of increased operating expenses.
# - Impact of future pandemics or disease outbreaks on the business and financial results.
# - Potential harm to the business due to economic conditions and their effect on discretionary consumer spending.
#
# 2021 Risk Factors:
# - Adverse impact of the COVID-19 pandemic and actions to mitigate it on the business.
# - Potential reclassification of drivers as employees instead of independent contractors.
# - Highly competitive nature of the mobility, delivery, and logistics industries.
# - Need to lower fares or service fees and offer incentives to remain competitive.
# - History of significant losses and uncertainty of achieving profitability.
# - Importance of attracting and maintaining a critical mass of platform users.
#
# 2020 Risk Factors:
# - Adverse impact of the COVID-19 pandemic on the business.
# - Potential reclassification of drivers as employees.
# - Highly competitive nature of the mobility, delivery, and logistics industries.
# - Need to lower fares or service fees to remain competitive.
# - History of significant losses and potential future expenses.
# - Importance of attracting and maintaining a critical mass of platform users.
# - Operational and cultural challenges faced by the company.
#
# 2019 Risk Factors:
# - Competition with local companies.
# - Differing levels of social acceptance.
# - Technological compatibility issues.
# - Exposure to improper business practices.
# - Legal uncertainty.
# - Difficulties in managing international operations.
# - Fluctuations in currency exchange rates.
# - Regulations governing local currencies.
# - Tax consequences.
# - Financial accounting burdens.
# - Difficulties in implementing financial systems.
# - Import and export restrictions.
# - Political and economic instability.
# - Public health concerns.
# - Reduced protection for intellectual property rights.
# - Limited influence over minority-owned affiliates.
# - Regulatory complexities.
#
# These comparisons highlight both common and unique risk factors that Uber faced in different years.

agent = OllamaAgent.from_tools(tools)  # verbose=False by default

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = agent.chat(text_input)
    print(f"Agent: {response}")

# User:  What were some of the legal proceedings against Uber in 2022?
# Agent: In 2022, Uber faced several legal proceedings. Some of the notable ones include:
#
# 1. Petition against Proposition 22: A petition was filed in California alleging that Proposition 22, which classifies app-based drivers as independent contractors, is unconstitutional.
#
# 2. Lawsuit by Massachusetts Attorney General: The Massachusetts Attorney General filed a lawsuit against Uber, claiming that drivers should be classified as employees and entitled to protections under wage and labor laws.
#
# 3. Allegations by New York Attorney General: The New York Attorney General made allegations against Uber regarding the misclassification of drivers and related employment violations.
#
# 4. Swiss social security rulings: Swiss social security rulings classified Uber drivers as employees, which could have implications for Uber's operations in Switzerland.
#
# 5. Class action lawsuits in Australia: Uber faced class action lawsuits in Australia, with allegations that the company conspired to harm participants in the taxi, hire-car, and limousine industries.
#
# It's important to note that the outcomes of these legal proceedings are uncertain and may vary.
#
# User:

logger.info("\n\n[DONE]", bright=True)
