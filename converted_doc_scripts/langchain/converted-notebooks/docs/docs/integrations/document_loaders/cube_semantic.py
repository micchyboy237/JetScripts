from jet.logger import logger
from langchain_community.document_loaders import CubeSemanticLoader
import jwt
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
# Cube Semantic Layer

This notebook demonstrates the process of retrieving Cube's data model metadata in a format suitable for passing to LLMs as embeddings, thereby enhancing contextual information.

### About Cube

[Cube](https://cube.dev/) is the Semantic Layer for building data apps. It helps data engineers and application developers access data from modern data stores, organize it into consistent definitions, and deliver it to every application.

Cube’s data model provides structure and definitions that are used as a context for LLM to understand data and generate correct queries. LLM doesn’t need to navigate complex joins and metrics calculations because Cube abstracts those and provides a simple interface that operates on the business-level terminology, instead of SQL table and column names. This simplification helps LLM to be less error-prone and avoid hallucinations.

### Example

**Input arguments (mandatory)**

`Cube Semantic Loader` requires 2 arguments:

- `cube_api_url`: The URL of your Cube's deployment REST API. Please refer to the [Cube documentation](https://cube.dev/docs/http-api/rest#configuration-base-path) for more information on configuring the base path.

- `cube_api_token`: The authentication token generated based on your Cube's API secret. Please refer to the [Cube documentation](https://cube.dev/docs/security#generating-json-web-tokens-jwt) for instructions on generating JSON Web Tokens (JWT).

**Input arguments (optional)**

- `load_dimension_values`: Whether to load dimension values for every string dimension or not.

- `dimension_values_limit`: Maximum number of dimension values to load.

- `dimension_values_max_retries`: Maximum number of retries to load dimension values.

- `dimension_values_retry_delay`: Delay between retries to load dimension values.
"""
logger.info("# Cube Semantic Layer")


api_url = "https://api-example.gcp-us-central1.cubecloudapp.dev/cubejs-api/v1/meta"
cubejs_api_secret = "api-secret-here"
security_context = {}
api_token = jwt.encode(security_context, cubejs_api_secret, algorithm="HS256")

loader = CubeSemanticLoader(api_url, api_token)

documents = loader.load()

"""
Returns a list of documents with the following attributes:

- `page_content`
- `metadata`
  - `table_name`
  - `column_name`
  - `column_data_type`
  - `column_title`
  - `column_description`
  - `column_values`
  - `cube_data_obj_type`
"""
logger.info("Returns a list of documents with the following attributes:")

page_content = "Users View City, None"

metadata = {
    "table_name": "users_view",
    "column_name": "users_view.city",
    "column_data_type": "string",
    "column_title": "Users View City",
    "column_description": "None",
    "column_member_type": "dimension",
    "column_values": [
        "Austin",
        "Chicago",
        "Los Angeles",
        "Mountain View",
        "New York",
        "Palo Alto",
        "San Francisco",
        "Seattle",
    ],
    "cube_data_obj_type": "view",
}

logger.info("\n\n[DONE]", bright=True)