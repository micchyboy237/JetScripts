from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import HumanMessage
from langchain_salesforce import SalesforceTool
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
# Salesforce

A tool for interacting with Salesforce CRM using LangChain.

## Overview

The `langchain-salesforce` package integrates LangChain with Salesforce CRM,
allowing you to query data, manage records, and explore object schemas
from LangChain applications.

### Key Features

- **SOQL Queries**: Execute Salesforce Object Query Language (SOQL) queries
- **Object Management**: Create, read, update, and delete (CRUD) operations on Salesforce objects  
- **Schema Exploration**: Describe object schemas and list available objects
- **Async Support**: Asynchronous operation support
- **Error Handling**: Detailed error messages
- **Environment Variable Support**: Load credentials from environment variables

## Setup

Install the required dependencies:
 
```bash
 pip install langchain-salesforce
 ```
 
## Authentication Setup

These environment variables will be automatically picked up by the integration.
 
## Getting Your Security Token
 
If you need a security token:
 1. Log into Salesforce
 2. Go to Settings
 3. Click on "Reset My Security Token" under "My Personal Information"
 4. Check your email for the new token
 
### Environment Variables (Recommended)
 
 Set up your Salesforce credentials as environment variables:
 
 ```bash
 export SALESFORCE_USERNAME="your-username@company.com"
 export SALESFORCE_PASSWORD="your-password"
 export SALESFORCE_SECURITY_TOKEN="your-security-token"
 export SALESFORCE_DOMAIN="login"  # Use "test" for sandbox environments
 ```

## Instantiation
"""
logger.info("# Salesforce")



username = os.getenv("SALESFORCE_USERNAME", "your-username")
password = os.getenv("SALESFORCE_PASSWORD", "your-password")
security_token = os.getenv("SALESFORCE_SECURITY_TOKEN", "your-security-token")
domain = os.getenv("SALESFORCE_DOMAIN", "login")

tool = SalesforceTool(
    username=username, password=password, security_token=security_token, domain=domain
)

"""
## Invocation
"""
logger.info("## Invocation")

def execute_salesforce_operation(
    operation, object_name=None, query=None, record_data=None, record_id=None
):
    """Executes a given Salesforce operation."""
    request = {"operation": operation}
    if object_name:
        request["object_name"] = object_name
    if query:
        request["query"] = query
    if record_data:
        request["record_data"] = record_data
    if record_id:
        request["record_id"] = record_id
    result = tool.invoke(request)
    return result

"""
## Query
This example queries Salesforce for 5 contacts.
"""
logger.info("## Query")

query_result = execute_salesforce_operation(
    operation="query", query="SELECT Id, Name, Email FROM Contact LIMIT 5"
)

"""
## Describe an Object
Fetches metadata for a specific Salesforce object.
"""
logger.info("## Describe an Object")

describe_result = execute_salesforce_operation(
    operation="describe", object_name="Account"
)

"""
## List Available Objects
Retrieves all objects available in the Salesforce instance.
"""
logger.info("## List Available Objects")

list_objects_result = execute_salesforce_operation(operation="list_objects")

"""
## Create a New Contact
Creates a new contact record in Salesforce.
"""
logger.info("## Create a New Contact")

create_result = execute_salesforce_operation(
    operation="create",
    object_name="Contact",
    record_data={"LastName": "Doe", "Email": "doe@example.com"},
)

"""
## Update a Contact
Updates an existing contact record.
"""
logger.info("## Update a Contact")

update_result = execute_salesforce_operation(
    operation="update",
    object_name="Contact",
    record_id="003XXXXXXXXXXXXXXX",
    record_data={"Email": "updated@example.com"},
)

"""
## Delete a Contact
Deletes a contact record from Salesforce.
"""
logger.info("## Delete a Contact")

delete_result = execute_salesforce_operation(
    operation="delete", object_name="Contact", record_id="003XXXXXXXXXXXXXXX"
)

"""
## Chaining
"""
logger.info("## Chaining")


tool = SalesforceTool(
    username=username, password=password, security_token=security_token, domain=domain
)

llm = ChatOllama(model="llama3.2")

contacts_query = {
    "operation": "query",
    "query": "SELECT Id, Name, Email, Phone FROM Contact LIMIT 3",
}

contacts_result = tool.invoke(contacts_query)

if contacts_result and "records" in contacts_result:
    contact_data = contacts_result["records"]

    analysis_prompt = f"""
    Please analyze the following Salesforce contact data and provide insights:

    Contact Data: {contact_data}

    Please provide:
    1. A summary of the contacts
    2. Any patterns you notice
    3. Suggestions for data quality improvements
    """

    message = HumanMessage(content=analysis_prompt)
    analysis_result = llm.invoke([message])

    logger.debug("\nLLM Analysis:")
    logger.debug(analysis_result.content)

"""
## API Reference

For comprehensive documentation and API reference, see:

- [langchain-salesforce README](https://github.com/colesmcintosh/langchain-salesforce/blob/main/README.md)
- [Simple Salesforce Documentation](https://simple-salesforce.readthedocs.io/en/latest/)

## Additional Resources

- [Salesforce SOQL Reference](https://developer.salesforce.com/docs/atlas.en-us.soql_sosl.meta/soql_sosl/)
- [Salesforce REST API Developer Guide](https://developer.salesforce.com/docs/atlas.en-us.api_rest.meta/api_rest/)
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)