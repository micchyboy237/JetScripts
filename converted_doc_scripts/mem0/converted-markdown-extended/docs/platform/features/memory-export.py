import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
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
title: Memory Export
description: 'Export memories in a structured format using customizable Pydantic schemas'
icon: "file-export"
iconType: "solid"
---

## Overview

The Memory Export feature allows you to create structured exports of memories using customizable Pydantic schemas. This process enables you to transform your stored memories into specific data formats that match your needs. You can apply various filters to narrow down which memories to export and define exactly how the data should be structured.

## Creating a Memory Export

To create a memory export, you'll need to:
1. Define your schema structure
2. Submit an export job
3. Retrieve the exported data

### Define Schema

Here's an example schema for extracting professional profile information:
"""
logger.info("## Overview")

{
    "$defs": {
        "EducationLevel": {
            "enum": ["high_school", "bachelors", "masters"],
            "title": "EducationLevel",
            "type": "string"
        },
        "EmploymentStatus": {
            "enum": ["full_time", "part_time", "student"],
            "title": "EmploymentStatus",
            "type": "string"
        }
    },
    "properties": {
        "full_name": {
            "anyOf": [
                {
                    "maxLength": 100,
                    "minLength": 2,
                    "type": "string"
                },
                {
                    "type": "null"
                }
            ],
            "default": null,
            "description": "The professional's full name",
            "title": "Full Name"
        },
        "current_role": {
            "anyOf": [
                {
                    "type": "string"
                },
                {
                    "type": "null"
                }
            ],
            "default": null,
            "description": "Current job title or role",
            "title": "Current Role"
        }
    },
    "title": "ProfessionalProfile",
    "type": "object"
}

"""
### Submit Export Job

You can optionally provide additional instructions to guide how memories are processed and structured during export using the `export_instructions` parameter.

<CodeGroup>
"""
logger.info("### Submit Export Job")

response = client.create_memory_export(
    schema=json_schema,
    user_id="alice"
)

export_instructions = """
1. Create a comprehensive profile with detailed information in each category
2. Only mark fields as "None" when absolutely no relevant information exists
3. Base all information directly on the user's memories
4. When contradictions exist, prioritize the most recent information
5. Clearly distinguish between factual statements and inferences
"""

filters = {
    "AND": [
        {"user_id": "alex"}
    ]
}

response = client.create_memory_export(
    schema=json_schema,
    filters=filters,
    export_instructions=export_instructions  # Optional
)

logger.debug(response)

"""

"""

async def run_async_code_2ab8444c():
    response = await client.createMemoryExport({
    return response
response = asyncio.run(run_async_code_2ab8444c())
logger.success(format_json(response))
    schema: json_schema,
    user_id: "alice"
})

export_instructions = `
1. Create a comprehensive profile with detailed information in each category
2. Only mark fields as "None" when absolutely no relevant information exists
3. Base all information directly on the user's memories
4. When contradictions exist, prioritize the most recent information
5. Clearly distinguish between factual statements and inferences
`

filters = {
    "AND": [
        {"user_id": "alex"}
    ]
}

async def run_async_code_9957e030():
    responseWithInstructions = await client.createMemoryExport({
    return responseWithInstructions
responseWithInstructions = asyncio.run(run_async_code_9957e030())
logger.success(format_json(responseWithInstructions))
    schema: json_schema,
    filters: filters,
    export_instructions: export_instructions
})

console.log(responseWithInstructions)

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/export/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "schema": {json_schema},
         "user_id": "alice",
         "export_instructions": "1. Create a comprehensive profile with detailed information\n2. Only mark fields as \"None\" when absolutely no relevant information exists"
     }'

"""

"""

{
    "message": "Memory export request received. The export will be ready in a few seconds.",
    "id": "550e8400-e29b-41d4-a716-446655440000"
}

"""
</CodeGroup>

### Retrieve Export

Once the export job is complete, you can retrieve the structured data in two ways:

#### Using Export ID

<CodeGroup>
"""
logger.info("### Retrieve Export")

response = client.get_memory_export(memory_export_id="550e8400-e29b-41d4-a716-446655440000")
logger.debug(response)

"""

"""

memory_export_id = "550e8400-e29b-41d4-a716-446655440000"

async def run_async_code_0e033058():
    response = await client.getMemoryExport({
    return response
response = asyncio.run(run_async_code_0e033058())
logger.success(format_json(response))
    memory_export_id: memory_export_id
})

console.log(response)

"""

"""

{
    "full_name": "John Doe",
    "current_role": "Senior Software Engineer",
    "years_experience": 8,
    "employment_status": "full_time",
    "education_level": "masters",
    "skills": ["Python", "AWS", "Machine Learning"]
}

"""
</CodeGroup>

#### Using Filters

<CodeGroup>
"""
logger.info("#### Using Filters")

filters = {
    "AND": [
        {"created_at": {"gte": "2024-07-10", "lte": "2024-07-20"}},
        {"user_id": "alex"}
    ]
}

response = client.get_memory_export(filters=filters)
logger.debug(response)

"""

"""

filters = {
    "AND": [
        {"created_at": {"gte": "2024-07-10", "lte": "2024-07-20"}},
        {"user_id": "alex"}
    ]
}

async def run_async_code_0e033058():
    response = await client.getMemoryExport({
    return response
response = asyncio.run(run_async_code_0e033058())
logger.success(format_json(response))
    filters: filters
})

console.log(response)

"""

"""

{
    "full_name": "John Doe",
    "current_role": "Senior Software Engineer",
    "years_experience": 8,
    "employment_status": "full_time",
    "education_level": "masters",
    "skills": ["Python", "AWS", "Machine Learning"]
}

"""
</CodeGroup>

## Available Filters

You can apply various filters to customize which memories are included in the export:

- `user_id`: Filter memories by specific user
- `agent_id`: Filter memories by specific agent
- `run_id`: Filter memories by specific run
- `session_id`: Filter memories by specific session
- `created_at`: Filter memories by date

<Note>
The export process may take some time to complete, especially when dealing with a large number of memories or complex schemas.
</Note>

If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Available Filters")

logger.info("\n\n[DONE]", bright=True)