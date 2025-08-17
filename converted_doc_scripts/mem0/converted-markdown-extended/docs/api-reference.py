import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import AsyncMemoryClient
from mem0 import MemoryClient
import asyncio
import os
import shutil
import { MemoryClient } from "mem0ai"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Overview
icon: "info"
iconType: "solid"
---

Mem0 provides a powerful set of APIs that allow you to integrate advanced memory management capabilities into your applications. Our APIs are designed to be intuitive, efficient, and scalable, enabling you to create, retrieve, update, and delete memories across various entities such as users, agents, apps, and runs.

## Key Features

- **Memory Management**: Add, retrieve, update, and delete memories with ease.
- **Entity-based Operations**: Perform operations on memories associated with specific users, agents, apps, or runs.
- **Advanced Search**: Utilize our search API to find relevant memories based on various criteria.
- **History Tracking**: Access the history of memory interactions for comprehensive analysis.
- **User Management**: Manage user entities and their associated memories.

## API Structure

Our API is organized into several main categories:

1. **Memory APIs**: Core operations for managing individual memories and collections.
2. **Entities APIs**: Manage different entity types (users, agents, etc.) and their associated memories.
3. **Search API**: Advanced search functionality to retrieve relevant memories.
4. **History API**: Track and retrieve the history of memory interactions.

## Authentication

All API requests require authentication using HTTP Basic Auth. Ensure you include your API key in the Authorization header of each request.

## Organizations and projects (optional)

Organizations and projects provide the following capabilities:

- **Multi-org/project Support**: Specify organization and project when initializing the Mem0 client to attribute API usage appropriately
- **Member Management**: Control access to data through organization and project membership
- **Access Control**: Only members can access memories and data within their organization/project scope
- **Team Isolation**: Maintain data separation between different teams and projects for secure collaboration

Example with the mem0 Python package:

<Tabs>
  <Tab title="Python">
"""
logger.info("## Key Features")

client = MemoryClient(org_id='YOUR_ORG_ID', project_id='YOUR_PROJECT_ID')

"""
</Tab>

  <Tab title="Node.js">
"""

client = new MemoryClient({organizationId: "YOUR_ORG_ID", projectId: "YOUR_PROJECT_ID"})

"""
</Tab>
</Tabs>

### Project Management Methods

The Mem0 client provides comprehensive project management capabilities through the `client.project` interface:

#### Get Project Details

Retrieve information about the current project:
"""
logger.info("### Project Management Methods")

project_info = client.project.get()

project_info = client.project.get(fields=["name", "description", "custom_categories"])

"""
#### Create a New Project

Create a new project within your organization:
"""
logger.info("#### Create a New Project")

new_project = client.project.create(
    name="My New Project",
    description="A project for managing customer support memories"
)

"""
#### Update Project Settings

Modify project configuration including custom instructions, categories, and graph settings:
"""
logger.info("#### Update Project Settings")

client.project.update(
    custom_categories=[
        {"customer_preferences": "Customer likes, dislikes, and preferences"},
        {"support_history": "Previous support interactions and resolutions"}
    ]
)

client.project.update(
    custom_instructions="..."
)

client.project.update(enable_graph=True)

client.project.update(
    custom_instructions="...",
    custom_categories=[
        {"personal_info": "User personal information and preferences"},
        {"work_context": "Professional context and work-related information"}
    ],
    enable_graph=True
)

"""
#### Delete Project

<Note>
This action will remove all memories, messages, and other related data in the project. This operation is irreversible.
</Note>

Remove a project and all its associated data:
"""
logger.info("#### Delete Project")

result = client.project.delete()

"""
#### Member Management

Manage project members and their access levels:
"""
logger.info("#### Member Management")

members = client.project.get_members()

client.project.add_member(
    email="colleague@company.com",
    role="READER"  # or "OWNER"
)

client.project.update_member(
    email="colleague@company.com",
    role="OWNER"
)

client.project.remove_member(email="colleague@company.com")

"""
#### Member Roles

- **READER**: Can view and search memories, but cannot modify project settings or manage members
- **OWNER**: Full access including project modification, member management, and all reader permissions

#### Async Support

All project methods are also available in async mode:
"""
logger.info("#### Member Roles")


async def manage_project():
    client = AsyncMemoryClient(org_id='YOUR_ORG_ID', project_id='YOUR_PROJECT_ID')

    async def run_async_code_a29d2b1a():
        async def run_async_code_30058363():
            project_info = await client.project.get()
            return project_info
        project_info = asyncio.run(run_async_code_30058363())
        logger.success(format_json(project_info))
        return project_info
    project_info = asyncio.run(run_async_code_a29d2b1a())
    logger.success(format_json(project_info))
    async def run_async_code_54caa456():
        await client.project.update(enable_graph=True)
        return 
     = asyncio.run(run_async_code_54caa456())
    logger.success(format_json())
    async def run_async_code_4dca5443():
        async def run_async_code_bbb64315():
            members = await client.project.get_members()
            return members
        members = asyncio.run(run_async_code_bbb64315())
        logger.success(format_json(members))
        return members
    members = asyncio.run(run_async_code_4dca5443())
    logger.success(format_json(members))

asyncio.run(manage_project())

"""
## Getting Started

To begin using the Mem0 API, you'll need to:

1. Sign up for a [Mem0 account](https://app.mem0.ai) and obtain your API key.
2. Familiarize yourself with the API endpoints and their functionalities.
3. Make your first API call to add or retrieve a memory.

Explore the detailed documentation for each API endpoint to learn more about request/response formats, parameters, and example usage.
"""
logger.info("## Getting Started")

logger.info("\n\n[DONE]", bright=True)