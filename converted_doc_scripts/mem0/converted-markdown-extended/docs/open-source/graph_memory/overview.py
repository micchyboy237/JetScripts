from jet.logger import CustomLogger
from mem0 import Memory
import os
import shutil
import { Memory } from "mem0ai/oss"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Overview
description: 'Enhance your memory system with graph-based knowledge representation and retrieval'
icon: "info"
iconType: "solid"
---

Mem0 now supports **Graph Memory**.
With Graph Memory, users can now create and utilize complex relationships between pieces of information, allowing for more nuanced and context-aware responses.
This integration enables users to leverage the strengths of both vector-based and graph-based approaches, resulting in more accurate and comprehensive information retrieval and generation.

<Note>
NodeSDK now supports Graph Memory. 🎉
</Note>

## Installation

To use Mem0 with Graph Memory support, install it using pip:

<CodeGroup>
"""
logger.info("## Installation")

pip install "mem0ai[graph]"

"""

"""

npm install mem0ai

"""
</CodeGroup>

This command installs Mem0 along with the necessary dependencies for graph functionality.

Try Graph Memory on Google Colab.
<a target="_blank" href="https://colab.research.google.com/drive/1PfIGVHnliIlG2v8cx0g45TF0US-jRPZ1?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


<iframe
width="100%"
height="400"
src="https://www.youtube.com/embed/u_ZAqNNVtXA"
title="YouTube video player"
frameborder="0"
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen
></iframe>

## Initialize Graph Memory

To initialize Graph Memory you'll need to set up your configuration with graph
store providers. Currently, we support [Neo4j](#initialize-neo4j), [Memgraph](#initialize-memgraph), [Neptune Analytics](#initialize-neptune-analytics), and [Kuzu](#initialize-kuzu) as graph store providers.


### Initialize Neo4j

You can setup [Neo4j](https://neo4j.com/) locally or use the hosted [Neo4j AuraDB](https://neo4j.com/product/auradb/).

<Note>If you are using Neo4j locally, then you need to install [APOC plugins](https://neo4j.com/labs/apoc/4.1/installation/).</Note>

User can also customize the LLM for Graph Memory from the [Supported LLM list](https://docs.mem0.ai/components/llms/overview) with three levels of configuration:

1. **Main Configuration**: If `llm` is set in the main config, it will be used for all graph operations.
2. **Graph Store Configuration**: If `llm` is set in the graph_store config, it will override the main config `llm` and be used specifically for graph operations.
3. **Default Configuration**: If no custom LLM is set, the default LLM (`gpt-4o-2024-08-06`) will be used for all graph operations.

Here's how you can do it:


<CodeGroup>
"""
logger.info("## Initialize Graph Memory")


config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://xxx",
            "username": "neo4j",
            "password": "xxx"
        }
    }
}

m = Memory.from_config(config_dict=config)

"""

"""


config = {
    enableGraph: true,
    graphStore: {
        provider: "neo4j",
        config: {
            url: "neo4j+s://xxx",
            username: "neo4j",
            password: "xxx",
        }
    }
}

memory = new Memory(config)

"""

"""

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://xxx",
            "username": "neo4j",
            "password": "xxx"
        },
        "llm" : {
            "provider": "openai",
            "config": {
                "model": "llama-3.2-3b-instruct",
                "temperature": 0.0,
            }
        }
    }
}

m = Memory.from_config(config_dict=config)

"""

"""

config = {
    llm: {
        provider: "openai",
        config: {
            model: "gpt-4o",
            temperature: 0.2,
            max_tokens: 2000,
        }
    },
    enableGraph: true,
    graphStore: {
        provider: "neo4j",
        config: {
            url: "neo4j+s://xxx",
            username: "neo4j",
            password: "xxx",
        },
        llm: {
            provider: "openai",
            config: {
                model: "llama-3.2-3b-instruct",
                temperature: 0.0,
            }
        }
    }
}

memory = new Memory(config)

"""
</CodeGroup>

<Note>
If you are using NodeSDK, you need to pass `enableGraph` as `true` in the `config` object.
</Note>

### Initialize Memgraph

Run Memgraph with Docker:
"""
logger.info("### Initialize Memgraph")

docker run -p 7687:7687 memgraph/memgraph-mage:latest --schema-info-enabled=True

"""
The `--schema-info-enabled` flag is set to `True` for more performant schema
generation.

Additional information can be found on [Memgraph
documentation](https://memgraph.com/docs).

User can also customize the LLM for Graph Memory from the [Supported LLM list](https://docs.mem0.ai/components/llms/overview) with three levels of configuration:

1. **Main Configuration**: If `llm` is set in the main config, it will be used for all graph operations.
2. **Graph Store Configuration**: If `llm` is set in the graph_store config, it will override the main config `llm` and be used specifically for graph operations.
3. **Default Configuration**: If no custom LLM is set, the default LLM (`gpt-4o-2024-08-06`) will be used for all graph operations.

Here's how you can do it:


<CodeGroup>
"""
logger.info("The `--schema-info-enabled` flag is set to `True` for more performant schema")


config = {
    "graph_store": {
        "provider": "memgraph",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "memgraph",
            "password": "xxx",
        },
    },
}

m = Memory.from_config(config_dict=config)

"""

"""

config = {
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-large", "embedding_dims": 1536},
    },
    "graph_store": {
        "provider": "memgraph",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "memgraph",
            "password": "xxx"
        }
    }
}

m = Memory.from_config(config_dict=config)

"""
</CodeGroup>

### Initialize Neptune Analytics

Mem0 now supports Amazon Neptune Analytics as a graph store provider. This integration allows you to use Neptune Analytics for storing and querying graph-based memories.

You can use Neptune Analytics as part of an Amazon tech stack [Setup AWS Bedrock, AOSS, and Neptune](https://docs.mem0.ai/examples/aws_example#aws-bedrock-and-aoss)

#### Instance Setup

Create an Amazon Neptune Analytics instance in your AWS account following the [AWS documentation](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/get-started.html).
- Public connectivity is not enabled by default, and if accessing from outside a VPC, it needs to be enabled.
- Once the Amazon Neptune Analytics instance is available, you will need the graph-identifier to connect.
- The Neptune Analytics instance must be created using the same vector dimensions as the embedding model creates. See: https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vector-index.html

#### Attach Credentials

 Configure your AWS credentials with access to your Amazon Neptune Analytics resources by following the [Configuration and credentials precedence](https://docs.aws.amazon.com/cli/v1/userguide/cli-chap-configure.html#configure-precedence).
- For example, add your SSH access key session token via environment variables:
"""
logger.info("### Initialize Neptune Analytics")

export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_SESSION_TOKEN=your-session-token
export AWS_DEFAULT_REGION=your-region

"""
- The IAM user or role making the request must have a policy attached that allows one of the following IAM actions in that neptune-graph:
  - neptune-graph:ReadDataViaQuery
  - neptune-graph:WriteDataViaQuery
  - neptune-graph:DeleteDataViaQuery

#### Usage

The Neptune memory store uses AWS LangChain Python API to connect to Neptune instances.  For additional configuration options for connecting to your Amazon Neptune Analytics instance see [AWS LangChain API documentation](https://python.langchain.com/api_reference/aws/graphs/langchain_aws.graphs.neptune_graph.NeptuneAnalyticsGraph.html).

<CodeGroup>
"""
logger.info("#### Usage")


config = {
    "graph_store": {
        "provider": "neptune",
        "config": {
            "endpoint": "neptune-graph://<GRAPH_ID>",
        },
    },
}

m = Memory.from_config(config_dict=config)

"""
</CodeGroup>

#### Troubleshooting

- For issues connecting to Amazon Neptune Analytics, please refer to the [Connecting to a graph guide](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/gettingStarted-connecting.html).

- For issues related to authentication, refer to the [boto3 client configuration options](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html).

- For more details on how to connect, configure, and use the graph_memory graph store, see the [Neptune Analytics example notebook](examples/graph-db-demo/neptune-analytics-example.ipynb).

### Initialize Kuzu

[Kuzu](https://kuzudb.com) is a fully local in-process graph database system that runs openCypher queries.
Kuzu comes embedded into the Python package and there is no additional setup required.

Kuzu needs a path to a file where it will store the graph database. For example:

<CodeGroup>
"""
logger.info("#### Troubleshooting")

config = {
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db": "/tmp/mem0-example.kuzu"
        }
    }
}

"""
</CodeGroup>

Kuzu can also store its database in memory. Note that in this mode, all stored memories will be lost
after the program has finished executing.

<CodeGroup>
"""
logger.info("Kuzu can also store its database in memory. Note that in this mode, all stored memories will be lost")

config = {
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db": ":memory:"
        }
    }
}

"""
</CodeGroup>

You can then use the above configuration in the usual way:

<CodeGroup>
"""
logger.info("You can then use the above configuration in the usual way:")


m = Memory.from_config(config_dict=config)

"""
</CodeGroup>

## Graph Operations
Mem0's graph memory supports the following operations:

### Add Memories

<Note>
Mem0 with Graph Memory supports "user_id", "agent_id", and "run_id" parameters. You can use any combination of these to organize your memories. Use "userId", "agentId", and "runId" in NodeSDK.
</Note>

<CodeGroup>
"""
logger.info("## Graph Operations")

m.add("I like pizza", user_id="alice")

m.add("I like pizza", user_id="alice", agent_id="food-assistant")

m.add("I like pizza", user_id="alice", agent_id="food-assistant", run_id="session-123")

"""

"""

memory.add("I like pizza", { userId: "alice" })

memory.add("I like pizza", { userId: "alice", agentId: "food-assistant" })

"""

"""

{'message': 'ok'}

"""
</CodeGroup>


### Get all memories

<CodeGroup>
"""
logger.info("### Get all memories")

m.get_all(user_id="alice")

m.get_all(user_id="alice", agent_id="food-assistant")

m.get_all(user_id="alice", run_id="session-123")

m.get_all(user_id="alice", agent_id="food-assistant", run_id="session-123")

"""

"""

memory.getAll({ userId: "alice" })

memory.getAll({ userId: "alice", agentId: "food-assistant" })

"""

"""

{
    'memories': [
        {
            'id': 'de69f426-0350-4101-9d0e-5055e34976a5',
            'memory': 'Likes pizza',
            'hash': '92128989705eef03ce31c462e198b47d',
            'metadata': None,
            'created_at': '2024-08-20T14:09:27.588719-07:00',
            'updated_at': None,
            'user_id': 'alice',
            'agent_id': 'food-assistant'
        }
    ],
    'entities': [
        {
            'source': 'alice',
            'relationship': 'likes',
            'target': 'pizza'
        }
    ]
}

"""
</CodeGroup>

### Search Memories

<CodeGroup>
"""
logger.info("### Search Memories")

m.search("tell me my name.", user_id="alice")

m.search("tell me my name.", user_id="alice", agent_id="food-assistant")

m.search("tell me my name.", user_id="alice", run_id="session-123")

m.search("tell me my name.", user_id="alice", agent_id="food-assistant", run_id="session-123")

"""

"""

memory.search("tell me my name.", { userId: "alice" })

memory.search("tell me my name.", { userId: "alice", agentId: "food-assistant" })

"""

"""

{
    'memories': [
        {
            'id': 'de69f426-0350-4101-9d0e-5055e34976a5',
            'memory': 'Likes pizza',
            'hash': '92128989705eef03ce31c462e198b47d',
            'metadata': None,
            'created_at': '2024-08-20T14:09:27.588719-07:00',
            'updated_at': None,
            'user_id': 'alice',
            'agent_id': 'food-assistant'
        }
    ],
    'entities': [
        {
            'source': 'alice',
            'relationship': 'likes',
            'target': 'pizza'
        }
    ]
}

"""
</CodeGroup>


### Delete all Memories

<CodeGroup>
"""
logger.info("### Delete all Memories")

m.delete_all(user_id="alice")

m.delete_all(user_id="alice", agent_id="food-assistant")

"""

"""

memory.deleteAll({ userId: "alice" })

memory.deleteAll({ userId: "alice", agentId: "food-assistant" })

"""
</CodeGroup>

# Example Usage
Here's an example of how to use Mem0's graph operations:

1. First, we'll add some memories for a user named Alice.
2. Then, we'll visualize how the graph evolves as we add more memories.
3. You'll see how entities and relationships are automatically extracted and connected in the graph.

### Add Memories

Below are the steps to add memories and visualize the graph:

<Steps>
  <Step title="Add memory 'I like going to hikes'">

<CodeGroup>
"""
logger.info("# Example Usage")

m.add("I like going to hikes", user_id="alice123")

"""

"""

memory.add("I like going to hikes", { userId: "alice123" })

"""
</CodeGroup>
![Graph Memory Visualization](/images/graph_memory/graph_example1.png)

</Step>
<Step title="Add memory 'I love to play badminton'">

<CodeGroup>
"""

m.add("I love to play badminton", user_id="alice123")

"""

"""

memory.add("I love to play badminton", { userId: "alice123" })

"""
</CodeGroup>

![Graph Memory Visualization](/images/graph_memory/graph_example2.png)

</Step>

<Step title="Add memory 'I hate playing badminton'">

<CodeGroup>
"""

m.add("I hate playing badminton", user_id="alice123")

"""

"""

memory.add("I hate playing badminton", { userId: "alice123" })

"""
</CodeGroup>

![Graph Memory Visualization](/images/graph_memory/graph_example3.png)

</Step>

<Step title="Add memory 'My friend name is john and john has a dog named tommy'">

<CodeGroup>
"""

m.add("My friend name is john and john has a dog named tommy", user_id="alice123")

"""

"""

memory.add("My friend name is john and john has a dog named tommy", { userId: "alice123" })

"""
</CodeGroup>

![Graph Memory Visualization](/images/graph_memory/graph_example4.png)

</Step>

<Step title="Add memory 'My name is Alice'">

<CodeGroup>
"""

m.add("My name is Alice", user_id="alice123")

"""

"""

memory.add("My name is Alice", { userId: "alice123" })

"""
</CodeGroup>

![Graph Memory Visualization](/images/graph_memory/graph_example5.png)

</Step>

<Step title="Add memory 'John loves to hike and Harry loves to hike as well'">

<CodeGroup>
"""

m.add("John loves to hike and Harry loves to hike as well", user_id="alice123")

"""

"""

memory.add("John loves to hike and Harry loves to hike as well", { userId: "alice123" })

"""
</CodeGroup>

![Graph Memory Visualization](/images/graph_memory/graph_example6.png)

</Step>

<Step title="Add memory 'My friend peter is the spiderman'">

<CodeGroup>
"""

m.add("My friend peter is the spiderman", user_id="alice123")

"""

"""

memory.add("My friend peter is the spiderman", { userId: "alice123" })

"""
</CodeGroup>

![Graph Memory Visualization](/images/graph_memory/graph_example7.png)

</Step>

</Steps>


### Search Memories

<CodeGroup>
"""
logger.info("### Search Memories")

m.search("What is my name?", user_id="alice123")

"""

"""

memory.search("What is my name?", { userId: "alice123" })

"""

"""

{
    'memories': [...],
    'entities': [
        {'source': 'alice123', 'relation': 'dislikes_playing','destination': 'badminton'},
        {'source': 'alice123', 'relation': 'friend', 'destination': 'peter'},
        {'source': 'alice123', 'relation': 'friend', 'destination': 'john'},
        {'source': 'alice123', 'relation': 'has_name', 'destination': 'alice'},
        {'source': 'alice123', 'relation': 'likes', 'destination': 'hiking'}
    ]
}

"""
</CodeGroup>

Below graph visualization shows what nodes and relationships are fetched from the graph for the provided query.

![Graph Memory Visualization](/images/graph_memory/graph_example8.png)

<CodeGroup>
"""
logger.info("Below graph visualization shows what nodes and relationships are fetched from the graph for the provided query.")

m.search("Who is spiderman?", user_id="alice123")

"""

"""

memory.search("Who is spiderman?", { userId: "alice123" })

"""

"""

{
    'memories': [...],
    'entities': [
        {'source': 'peter', 'relation': 'identity','destination': 'spiderman'}
    ]
}

"""
</CodeGroup>

![Graph Memory Visualization](/images/graph_memory/graph_example9.png)

> **Note:** The Graph Memory implementation is not standalone. You will be adding/retrieving memories to the vector store and the graph store simultaneously.

## Using Multiple Agents with Graph Memory


When working with multiple agents and sessions, you can use the "agent_id" and "run_id" parameters to organize memories by user, agent, and run context. This allows you to:

1. Create agent-specific knowledge graphs
2. Share common knowledge between agents
3. Isolate sensitive or specialized information to specific agents
4. Track conversation sessions and runs separately
5. Maintain context across different execution contexts

### Example: Multi-Agent Setup

<CodeGroup>
"""
logger.info("## Using Multiple Agents with Graph Memory")

m.add("I prefer Italian cuisine", user_id="bob", agent_id="food-assistant")
m.add("I'm allergic to peanuts", user_id="bob", agent_id="health-assistant")
m.add("I live in Seattle", user_id="bob")  # Shared across all agents

m.add("Current session: discussing dinner plans", user_id="bob", agent_id="food-assistant", run_id="dinner-session-001")
m.add("Previous session: allergy consultation", user_id="bob", agent_id="health-assistant", run_id="health-session-001")

food_preferences = m.search("What food do I like?", user_id="bob", agent_id="food-assistant")
health_info = m.search("What are my allergies?", user_id="bob", agent_id="health-assistant")
location = m.search("Where do I live?", user_id="bob")  # Searches across all agents

current_session = m.search("What are we discussing?", user_id="bob", run_id="dinner-session-001")

"""

"""

memory.add("I prefer Italian cuisine", { userId: "bob", agentId: "food-assistant" })
memory.add("I'm allergic to peanuts", { userId: "bob", agentId: "health-assistant" })
memory.add("I live in Seattle", { userId: "bob" });  // Shared across all agents

foodPreferences = memory.search("What food do I like?", { userId: "bob", agentId: "food-assistant" })
healthInfo = memory.search("What are my allergies?", { userId: "bob", agentId: "health-assistant" })
location = memory.search("Where do I live?", { userId: "bob" });  // Searches across all agents

"""
</CodeGroup>

If you want to use a managed version of Mem0, please check out [Mem0](https://mem0.dev/pd). If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("If you want to use a managed version of Mem0, please check out [Mem0](https://mem0.dev/pd). If you have any questions, please feel free to reach out to us using one of the following methods:")

logger.info("\n\n[DONE]", bright=True)