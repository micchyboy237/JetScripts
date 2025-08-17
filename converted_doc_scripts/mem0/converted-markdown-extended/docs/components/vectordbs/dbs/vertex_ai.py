from jet.logger import CustomLogger
from mem0 import Memory
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Vertex AI Vector Search
---


### Usage

To use Google Cloud Vertex AI Vector Search with `mem0`, you need to configure the `vector_store` in your `mem0` config:
"""
logger.info("### Usage")


os.environ["GOOGLE_API_KEY"] = "sk-xx"

config = {
    "vector_store": {
        "provider": "vertex_ai_vector_search",
        "config": {
            "endpoint_id": "YOUR_ENDPOINT_ID",            # Required: Vector Search endpoint ID
            "index_id": "YOUR_INDEX_ID",                  # Required: Vector Search index ID
            "deployment_index_id": "YOUR_DEPLOYMENT_INDEX_ID",  # Required: Deployment-specific ID
            "project_id": "YOUR_PROJECT_ID",              # Required: Google Cloud project ID
            "project_number": "YOUR_PROJECT_NUMBER",      # Required: Google Cloud project number
            "region": "YOUR_REGION",                      # Optional: Defaults to GOOGLE_CLOUD_REGION
            "credentials_path": "path/to/credentials.json", # Optional: Defaults to GOOGLE_APPLICATION_CREDENTIALS
            "vector_search_api_endpoint": "YOUR_API_ENDPOINT" # Required for get operations
        }
    }
}
m = Memory.from_config(config)
m.add("Your text here", user_id="user", metadata={"category": "example"})

"""
### Required Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `endpoint_id` | Vector Search endpoint ID | Yes |
| `index_id` | Vector Search index ID | Yes |
| `deployment_index_id` | Deployment-specific index ID | Yes |
| `project_id` | Google Cloud project ID | Yes |
| `project_number` | Google Cloud project number | Yes |
| `vector_search_api_endpoint` | Vector search API endpoint | Yes (for get operations) |
| `region` | Google Cloud region | No (defaults to GOOGLE_CLOUD_REGION) |
| `credentials_path` | Path to service account credentials | No (defaults to GOOGLE_APPLICATION_CREDENTIALS) |
"""
logger.info("### Required Parameters")

logger.info("\n\n[DONE]", bright=True)