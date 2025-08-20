import asyncio
from jet.transformers.formatters import format_json
from datetime import date
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_cloud_services.extract import (
    ExtractConfig,
    ExtractMode,
    LlamaExtract,
    SourceText,
)
from llama_cloud_services.parse import LlamaParse
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.core.workflow import (
    Workflow,
    Event,
    Context,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from neo4j import AsyncGraphDatabase
from openai import AsyncMLX
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import re
import shutil
import uuid


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Building a Knowledge Graph with LlamaCloud & Neo4J

RAG is as a powerful technique for enhancing LLMs with external knowledge, but traditional semantic similarity search often fails to capture nuanced relationships between entities, and can miss critical context that spans across multiple documents. By transforming unstructured documents into structured knowledge representations, we can perform complex graph traversals, relationship queries, and contextual reasoning that goes far beyond simple similarity matching.

Tools like LlamaParse and LlamaExtract provide robust parsing and extraction capabilities to convert raw documents into structured data, while Neo4j serves as the backbone for knowledge graph representation, forming the foundation of GraphRAG architectures that can understand not just what information exists, but how it all connects together.

In this end-to-end tutorial, we will walk through an example of legal document processing that showcases the full pipeline shown below.

The pipeline contains the following steps:
- Use [LlamaParse](https://www.llamaindex.ai/llamaparse) to parse PDF documents and extract readable text
- Employ a large language model to classify the contract type, enabling context-aware processing
- Leverage [LlamaExtract](https://www.llamaindex.ai/llamaextract) to extract different sets of relevant attributes tailored to each specific contract category based on the classification
- Store all structured information into a Neo4j knowledge graph, creating a rich, queryable representation that captures both content and intricate relationships within legal documents

## Setting Up Requirements
"""
logger.info("# Building a Knowledge Graph with LlamaCloud & Neo4J")

# !pip install llama-index-workflows llama-cloud-services jsonschema openai neo4j llama-index-llms-ollama


# from getpass import getpass


"""
# Download Sample Contract

Here, we download a sample PDF from the Cuad dataset
"""
logger.info("# Download Sample Contract")

# !wget https://raw.githubusercontent.com/tomasonjo/blog-datasets/5e3939d10216b7663687217c1646c30eb921d92f/CybergyHoldingsInc_Affliate%20Agreement.pdf

"""
## Set Up Neo4J

For Neo4j, the simplest approach is to create a free [Aura database instance](https://neo4j.com/product/auradb/), and copy your credentials here.
"""
logger.info("## Set Up Neo4J")

db_url = "your-db-url"
username = "neo4j"
password = "your-password"

neo4j_driver = AsyncGraphDatabase.driver(
    db_url,
    auth=(
        username,
        password,
    ),
)

"""
## Parse the Contract with LlamaParse

Next, we set up LlamaParse and parse the PDF. In this case, we're using `parse_page_without_llm` mode.
"""
logger.info("## Parse the Contract with LlamaParse")


# os.environ["LLAMA_CLOUD_API_KEY"] = getpass("Enter your Llama API key: ")
# os.environ["OPENAI_API_KEY"] = getpass("Enter your MLX API key: ")

parser = LlamaParse(parse_mode="parse_page_without_llm")

pdf_path = "CybergyHoldingsInc_Affliate Agreement.pdf"


async def run_async_code_bf48884c():
    async def run_async_code_e1e182e5():
        results = await parser.aparse(pdf_path)
        return results
    results = asyncio.run(run_async_code_e1e182e5())
    logger.success(format_json(results))
    return results
results = asyncio.run(run_async_code_bf48884c())
logger.success(format_json(results))

logger.debug(results.pages[0].text)

"""
## Contract classification

In this example, we want to classify incoming contracts. They can either be `Affiliate Agreements` or `Co Branding`. We define a classification prompt below, and ask the LLM to return the reason for the classification as well.
"""
logger.info("## Contract classification")

llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit")

classification_prompt = """You are a legal document classification assistant.
Your task is to identify the most likely contract type based on the content of the first 10 pages of a contract.

Instructions:

Read the contract excerpt below (up to the first 10 pages).

Review the list of possible contract types.

Choose the single most appropriate contract type from the list.

Justify your classification briefly, based only on the information in the excerpt.

Contract Excerpt:
{contract_text}

Possible Contract Types:
{contract_type_list}

Output Format:
<Reason>brief_justification</Reason>
<ContractType>chosen_type_from_list</ContractType>
"""


def extract_reason_and_contract_type(text):
    reason_match = re.search(r"<Reason>(.*?)</Reason>", text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else None

    contract_type_match = re.search(
        r"<ContractType>(.*?)</ContractType>", text, re.DOTALL
    )
    contract_type = (
        contract_type_match.group(1).strip() if contract_type_match else None
    )

    return {"reason": reason, "contract_type": contract_type}


async def classify_contract(
    contract_text: str, contract_types: list[str]
) -> dict:
    prompt = classification_prompt.format(
        contract_text=file_content, contract_type_list=contract_types
    )
    history = [ChatMessage(role="user", content=prompt)]

    async def run_async_code_555ad4e9():
        async def run_async_code_9edc3f04():
            response = llm.chat(history)
            return response
        response = asyncio.run(run_async_code_9edc3f04())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_555ad4e9())
    logger.success(format_json(response))
    return extract_reason_and_contract_type(response.message.content)

contract_types = ["Affiliate_Agreements", "Co_Branding"]

file_content = " ".join([el.text for el in results.pages[:10]])


async def run_async_code_e3fa66cd():
    async def run_async_code_89a28349():
        classification = await classify_contract(file_content, contract_types)
        return classification
    classification = asyncio.run(run_async_code_89a28349())
    logger.success(format_json(classification))
    return classification
classification = asyncio.run(run_async_code_e3fa66cd())
logger.success(format_json(classification))
classification

"""
## Setting Up LlamaExtract

Next, we define some schemas which we can use to extract relevant information from our contracts with. The fields we define are a mix of summarization and structured data extraction.

Here we define two Pydantic models: `Location` captures structured address information with optional fields for country, state, and address, while `Party` represents contract parties with a required name and optional location details. The Field descriptions help guide the extraction process by telling the LLM exactly what information to look for in each field.
"""
logger.info("## Setting Up LlamaExtract")


class Location(BaseModel):
    """Location information with structured address components."""

    country: Optional[str] = Field(None, description="Country")
    state: Optional[str] = Field(None, description="State or province")
    address: Optional[str] = Field(None, description="Street address or city")


class Party(BaseModel):
    """Party information with name and location."""

    name: str = Field(description="Party name")
    location: Optional[Location] = Field(
        None, description="Party location details"
    )


"""
Remember we have multiple contract types, so we need to define specific extraction schemas for each type and create a mapping system to dynamically select the appropriate schema based on our classification result.
"""
logger.info("Remember we have multiple contract types, so we need to define specific extraction schemas for each type and create a mapping system to dynamically select the appropriate schema based on our classification result.")


class BaseContract(BaseModel):
    """Base contract class with common fields."""

    parties: Optional[List[Party]] = Field(
        None, description="All contracting parties"
    )
    agreement_date: Optional[str] = Field(
        None, description="Contract signing date. Use YYYY-MM-DD"
    )
    effective_date: Optional[str] = Field(
        None, description="When contract becomes effective. Use YYYY-MM-DD"
    )
    expiration_date: Optional[str] = Field(
        None, description="Contract expiration date. Use YYYY-MM-DD"
    )
    governing_law: Optional[str] = Field(
        None, description="Governing jurisdiction"
    )
    termination_for_convenience: Optional[bool] = Field(
        None, description="Can terminate without cause"
    )
    anti_assignment: Optional[bool] = Field(
        None, description="Restricts assignment to third parties"
    )
    cap_on_liability: Optional[str] = Field(
        None, description="Liability limit amount"
    )


class AffiliateAgreement(BaseContract):
    """Affiliate Agreement extraction."""

    exclusivity: Optional[str] = Field(
        None, description="Exclusive territory or market rights"
    )
    non_compete: Optional[str] = Field(
        None, description="Non-compete restrictions"
    )
    revenue_profit_sharing: Optional[str] = Field(
        None, description="Commission or revenue split"
    )
    minimum_commitment: Optional[str] = Field(
        None, description="Minimum sales targets"
    )


class CoBrandingAgreement(BaseContract):
    """Co-Branding Agreement extraction."""

    exclusivity: Optional[str] = Field(
        None, description="Exclusive co-branding rights"
    )
    ip_ownership_assignment: Optional[str] = Field(
        None, description="IP ownership allocation"
    )
    license_grant: Optional[str] = Field(
        None, description="Brand/trademark licenses"
    )
    revenue_profit_sharing: Optional[str] = Field(
        None, description="Revenue sharing terms"
    )


mapping = {
    "Affiliate_Agreements": AffiliateAgreement,
    "Co_Branding": CoBrandingAgreement,
}

extractor = LlamaExtract()

agent = extractor.create_agent(
    name=f"extraction_workflow_import_{uuid.uuid4()}",
    data_schema=mapping[classification["contract_type"]],
    config=ExtractConfig(
        extraction_mode=ExtractMode.BALANCED,
    ),
)


async def async_func_78():
    result = await agent.aextract(
        files=SourceText(
            text_content=" ".join([el.text for el in results.pages]),
            filename=pdf_path,
        ),
    )
    return result
result = asyncio.run(async_func_78())
logger.success(format_json(result))

result.data

"""
## Import into Neo4j

The final step is to take our extracted structured information and build a knowledge graph that represents the relationships between contract entities. We need to define a graph model that specifies how our contract data should be organized as nodes and relationships in Neo4j.



Our graph model consists of three main node types:
- **Contract nodes** store the core agreement information including dates, terms, and legal clauses
- **Party nodes** represent the contracting entities with their names
- **Location nodes** capture geographic information with address components.

Now we'll import our extracted contract data into Neo4j according to our defined graph model.
"""
logger.info("## Import into Neo4j")

import_query = """
WITH $contract AS contract
MERGE (c:Contract {path: $path})
SET c += apoc.map.clean(contract, ["parties", "agreement_date", "effective_date", "expiration_date"], [])
// Cast to date
SET c.agreement_date = date(contract.agreement_date),
    c.effective_date = date(contract.effective_date),
    c.expiration_date = date(contract.expiration_date)

// Create parties with their locations
WITH c, contract
UNWIND coalesce(contract.parties, []) AS party
MERGE (p:Party {name: party.name})
MERGE (c)-[:HAS_PARTY]->(p)

// Create location nodes and link to parties
WITH p, party
WHERE party.location IS NOT NULL
MERGE (p)-[:HAS_LOCATION]->(l:Location)
SET l += party.location
"""


async def async_func_22():
    response = await neo4j_driver.execute_query(
        import_query, contract=result.data, path=pdf_path
    )
    return response
response = asyncio.run(async_func_22())
logger.success(format_json(response))
response.summary.counters

"""
## Bringing it All Together in a Workflow

Finally, we can combine all of this logic into one single executable agentic workflow. Let's make it so that the workflow can run by accepting a single PDF, adding new entries to our Neo4j graph each time.
"""
logger.info("## Bringing it All Together in a Workflow")

affiliage_extraction_agent = extractor.create_agent(
    name="Affiliate_Extraction",
    data_schema=AffiliateAgreement,
    config=ExtractConfig(
        extraction_mode=ExtractMode.BALANCED,
    ),
)
cobranding_extraction_agent = extractor.create_agent(
    name="CoBranding_Extraction",
    data_schema=CoBrandingAgreement,
    config=ExtractConfig(
        extraction_mode=ExtractMode.BALANCED,
    ),
)


class ClassifyDocEvent(Event):
    parsed_doc: str
    pdf_path: str


class ExtactAffiliate(Event):
    file_path: str


class ExtractCoBranding(Event):
    file_path: str


class BuildGraph(Event):
    file_path: str
    data: dict


class KnowledgeGraphBuilder(Workflow):
    def __init__(
        self,
        parser: LlamaParse,
        affiliate_extract_agent: LlamaExtract,
        branding_extract_agent: LlamaExtract,
        classification_prompt: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.parser = parser
        self.affiliate_extract_agent = affiliate_extract_agent
        self.branding_extract_agent = branding_extract_agent
        self.classification_prompt = classification_prompt
        self.llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit")

    @step
    async def parse_file(
        self, ctx: Context, ev: StartEvent
    ) -> ClassifyDocEvent:
        async def run_async_code_b11ae286():
            async def run_async_code_350bc292():
                results = await self.parser.aparse(ev.pdf_path)
                return results
            results = asyncio.run(run_async_code_350bc292())
            logger.success(format_json(results))
            return results
        results = asyncio.run(run_async_code_b11ae286())
        logger.success(format_json(results))
        parsed_doc = " ".join([el.text for el in results.pages[:10]])
        return ClassifyDocEvent(parsed_doc=parsed_doc, pdf_path=ev.pdf_path)

    @step
    async def classify_contract(
        self, ctx: Context, ev: ClassifyDocEvent
    ) -> ExtactAffiliate | ExtractCoBranding | StopEvent:
        prompt = self.classification_prompt.format(
            contract_text=ev.parsed_doc,
            contract_type_list=["Affiliate_Agreements", "Co_Branding"],
        )
        history = [ChatMessage(role="user", content=prompt)]

        async def run_async_code_9edc3f04():
            async def run_async_code_639360cb():
                response = llm.chat(history)
                return response
            response = asyncio.run(run_async_code_639360cb())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_9edc3f04())
        logger.success(format_json(response))
        reason_match = re.search(
            r"<Reason>(.*?)</Reason>", response.message.content, re.DOTALL
        )
        reason = reason_match.group(1).strip() if reason_match else None

        contract_type_match = re.search(
            r"<ContractType>(.*?)</ContractType>",
            response.message.content,
            re.DOTALL,
        )
        contract_type = (
            contract_type_match.group(1).strip()
            if contract_type_match
            else None
        )
        if contract_type == "Affiliate_Agreements":
            return ExtactAffiliate(file_path=ev.pdf_path)
        elif contract_type == "Co_Branding":
            return ExtractCoBranding(file_path=ev.pdf_path)
        else:
            return StopEvent()

    @step
    async def extract_affiliate(
        self, ctx: Context, ev: ExtactAffiliate
    ) -> BuildGraph:
        async def run_async_code_c49913c4():
            async def run_async_code_af9e8fcd():
                result = await self.affiliate_extract_agent.aextract(ev.file_path)
                return result
            result = asyncio.run(run_async_code_af9e8fcd())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_c49913c4())
        logger.success(format_json(result))
        return BuildGraph(data=result.data, file_path=ev.file_path)

    @step
    async def extract_co_branding(
        self, ctx: Context, ev: ExtractCoBranding
    ) -> BuildGraph:
        async def run_async_code_3b41763c():
            async def run_async_code_90a17065():
                result = await self.branding_extract_agent.aextract(ev.file_path)
                return result
            result = asyncio.run(run_async_code_90a17065())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_3b41763c())
        logger.success(format_json(result))
        return BuildGraph(data=result.data, file_path=ev.file_path)

    @step
    async def build_graph(self, ctx: Context, ev: BuildGraph) -> StopEvent:
        import_query = """
    WITH $contract AS contract
    MERGE (c:Contract {path: $path})
    SET c += apoc.map.clean(contract, ["parties", "agreement_date", "effective_date", "expiration_date"], [])
    // Cast to date
    SET c.agreement_date = date(contract.agreement_date),
      c.effective_date = date(contract.effective_date),
      c.expiration_date = date(contract.expiration_date)

    // Create parties with their locations
    WITH c, contract
    UNWIND coalesce(contract.parties, []) AS party
    MERGE (p:Party {name: party.name})
    MERGE (c)-[:HAS_PARTY]->(p)

    // Create location nodes and link to parties
    WITH p, party
    WHERE party.location IS NOT NULL
    MERGE (p)-[:HAS_LOCATION]->(l:Location)
    SET l += party.location
    """

        async def async_func_137():
            response = await neo4j_driver.execute_query(
                import_query, contract=ev.data, path=ev.file_path
            )
            return response
        response = asyncio.run(async_func_137())
        logger.success(format_json(response))
        return StopEvent(response.summary.counters)


knowledge_graph_builder = KnowledgeGraphBuilder(
    parser=parser,
    affiliate_extract_agent=affiliage_extraction_agent,
    branding_extract_agent=cobranding_extraction_agent,
    classification_prompt=classification_prompt,
    timeout=None,
    verbose=True,
)


async def async_func_151():
    response = await knowledge_graph_builder.run(
        pdf_path="CybergyHoldingsInc_Affliate Agreement.pdf"
    )
    return response
response = asyncio.run(async_func_151())
logger.success(format_json(response))

logger.info("\n\n[DONE]", bright=True)
