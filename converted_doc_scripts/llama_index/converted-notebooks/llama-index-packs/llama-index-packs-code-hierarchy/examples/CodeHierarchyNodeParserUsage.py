import asyncio
from jet.transformers.formatters import format_json
from IPython.display import Markdown, display
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.schema import NodeRelationship
from llama_index.core.text_splitter import CodeSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context
from llama_index.packs.code_hierarchy import CodeHierarchyAgentPack
from llama_index.packs.code_hierarchy import CodeHierarchyKeywordQueryEngine
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser
from pathlib import Path
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

# %load_ext autoreload
# %autoreload 2

"""
# Code Hierarchy Node Parser

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-code-hierarchy/examples/CodeHierarchyNodeParserUsage.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

The `CodeHierarchyNodeParser` is useful to split long code files into more reasonable chunks. What this will do is create a "Hierarchy" of sorts, where sections of the code are made more reasonable by replacing the scope body with short comments telling the LLM to search for a referenced node if it wants to read that context body. This is called skeletonization, and is toggled by setting `skeleton` to `True` which it is by default. Nodes in this hierarchy will be split based on scope, like function, class, or method scope, and will have links to their children and parents so the LLM can traverse the tree.

This notebook gives an initial demo of the pack, and then dives into a deeper technical exploration of how it works.

**NOTE:** Currently, this pack is configured to only work with `MLX` LLMs. But feel free to copy/download the source code and edit as needed!

## Installation and Import

First be sure to install the necessary [tree-sitter](https://tree-sitter.github.io/tree-sitter/) libraries.
"""
logger.info("# Code Hierarchy Node Parser")

# !pip install llama-index-packs-code-hierarchy llama-index




def print_python(python_text):
    """This function prints python text in ipynb nicely formatted."""
    display(Markdown("```python\n" + python_text + "```"))


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
## Initial Demo

First, lets run the pack by using nodes from the included `CodeHierarchyNodeParser`, and from there, explore further how it actually works.
"""
logger.info("## Initial Demo")

llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.2)

documents = SimpleDirectoryReader(
    input_files=[Path("../llama_index/packs/code_hierarchy/code_hierarchy.py")],
    file_metadata=lambda x: {"filepath": x},
).load_data()

split_nodes = CodeHierarchyNodeParser(
    language="python",
    code_splitter=CodeSplitter(language="python", max_chars=1000, chunk_lines=10),
).get_nodes_from_documents(documents)

pack = CodeHierarchyAgentPack(split_nodes=split_nodes, llm=llm)

logger.debug(
    pack.run(
        "How does the get_code_hierarchy_from_nodes function from the code hierarchy node parser work? Provide specific implementation details."
    )
)

"""
We can see that the agent explored the hierarchy of the code by requesting specific function names and IDs, in order to provide a full explanation of how the function works!

## Technical Explanations/Exploration

### Prepare your Data

Choose a directory you want to scan, and glob for all the code files you want to import.

In this case I'm going to glob all "*.py" files in the `llama_index/node_parser` directory.
"""
logger.info("## Technical Explanations/Exploration")

documents = SimpleDirectoryReader(
    input_files=[Path("../llama_index/packs/code_hierarchy/code_hierarchy.py")],
    file_metadata=lambda x: {"filepath": x},
).load_data()

split_nodes = CodeHierarchyNodeParser(
    language="python",
    code_splitter=CodeSplitter(language="python", max_chars=1000, chunk_lines=10),
).get_nodes_from_documents(documents)

"""
This should be the code hierarchy node parser itself. Lets have it parse itself!
"""
logger.info("This should be the code hierarchy node parser itself. Lets have it parse itself!")

logger.debug(f"Length of text: {len(documents[0].text)}")
print_python(documents[0].text[:1500] + "\n\n# ...")

"""
This is way too long to fit into the context of our LLM. So what are we to do? Well we will split it. We are going to use the `CodeHierarchyNodeParser` to split the nodes into more reasonable chunks.
"""
logger.info("This is way too long to fit into the context of our LLM. So what are we to do? Well we will split it. We are going to use the `CodeHierarchyNodeParser` to split the nodes into more reasonable chunks.")

split_nodes = CodeHierarchyNodeParser(
    language="python",
    code_splitter=CodeSplitter(language="python", max_chars=1000, chunk_lines=10),
).get_nodes_from_documents(documents)
logger.debug("Number of nodes after splitting:", len(split_nodes))

"""
Great! So that split up our data from 1 node into quite a few nodes! Whats the max length of any of these nodes?
"""
logger.info("Great! So that split up our data from 1 node into quite a few nodes! Whats the max length of any of these nodes?")

logger.debug(f"Longest text in nodes: {max(len(n.text) for n in split_nodes)}")

"""
That's much shorter than before! Let's look at a sample.
"""
logger.info("That's much shorter than before! Let's look at a sample.")

print_python(split_nodes[0].text)

"""
Without even needing a long printout we can see everything this module imported in the first document (which is at the module level) and some classes it defines.

We also see that it has put comments in place of code that was removed to make the text size more reasonable.
These can appear at the beginning or end of a chunk, or at a new scope level, like a class or function declaration.

`# Code replaced for brevity. See node_id {node_id}`

### Code Hierarchy

These scopes can be listed by the `CodeHierarchyNodeParser`, giving a "repo map" of sorts.
The namesake of this node parser, it creates a tree of scope names to use to search the code.
"""
logger.info("### Code Hierarchy")

logger.debug(CodeHierarchyNodeParser.get_code_hierarchy_from_nodes(split_nodes))

"""
### Exploration by the Programmer

So that we understand what is going on under the hood, what if we go to that node_id we found above?
"""
logger.info("### Exploration by the Programmer")

split_nodes_by_id = {n.node_id: n for n in split_nodes}
uuid_from_text = split_nodes[9].text.splitlines()[-1].split(" ")[-1]
logger.debug("Going to print the node with UUID:", uuid_from_text)
print_python(split_nodes_by_id[uuid_from_text].text)

"""
This is the next split in the file. It is prepended with the node before it and appended with the node after it as a comment.

We can also see the relationships on this node programmatically.
"""
logger.info("This is the next split in the file. It is prepended with the node before it and appended with the node after it as a comment.")

split_nodes_by_id[uuid_from_text].relationships

"""
The `NEXT` `PREV` relationships come from the `CodeSplitter` which is a component of the `CodeHierarchyNodeParser`. It is responsible for cutting up the nodes into chunks that are a certain character length. For more information about the `CodeSplitter` read this:

[Code Splitter](https://docs.llamaindex.ai/en/latest/api/llama_index.node_parser.CodeSplitter.html)

The `PARENT` and `CHILD` relationships come from the `CodeHierarchyNodeParser` which is responsible for creating the hierarchy of nodes. Things like classes, functions, and methods are nodes in this hierarchy.

The `SOURCE` is the original file that this node came from.
"""
logger.info("The `NEXT` `PREV` relationships come from the `CodeSplitter` which is a component of the `CodeHierarchyNodeParser`. It is responsible for cutting up the nodes into chunks that are a certain character length. For more information about the `CodeSplitter` read this:")


node_id = uuid_from_text
if NodeRelationship.NEXT not in split_nodes_by_id[node_id].relationships:
    logger.debug("No next node found!")
else:
    next_node_relationship_info = split_nodes_by_id[node_id].relationships[
        NodeRelationship.NEXT
    ]
    next_node = split_nodes_by_id[next_node_relationship_info.node_id]
    print_python(next_node.text)

"""
### Keyword Table and Usage by the LLM

Lets explore the use of this node parser in an index. We will be able to use any index which allows search by keyword, which should enable us to search for any node by it's uuid, or by any scope name.

We have created a `CodeHierarchyKeywordQueryEngine` which will allow us to search for nodes by their uuid, or by their scope name. It's `.query` method can be used as a simple search tool for any LLM. Given the repo map we created earlier, or the text of a split file, the LLM should be able to figure out what to search for very naturally.

Lets create the KeywordQueryEngine
"""
logger.info("### Keyword Table and Usage by the LLM")


query_engine = CodeHierarchyKeywordQueryEngine(
    nodes=split_nodes,
)

"""
Now we can get the same code as before.
"""
logger.info("Now we can get the same code as before.")

print_python(query_engine.query(split_nodes[0].node_id).response)

"""
But now we can also search for any node by it's common sense name.

For example, the class `_SignatureCaptureOptions` is a node in the hierarchy. We can search for it by name.

The reason we aren't getting more detail is because our min_characters is too low, try to increase it for more detail for any individual query.
"""
logger.info("But now we can also search for any node by it's common sense name.")

print_python(query_engine.query("_SignatureCaptureOptions").response)

"""
And by module name, in case the LLM sees something in an import statement and wants to know more about it.
"""
logger.info("And by module name, in case the LLM sees something in an import statement and wants to know more about it.")

print_python(query_engine.query("code_hierarchy").response)

"""
### As an Agent

We can convert the query engine to be used as a tool for an agent!
"""
logger.info("### As an Agent")


tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="code_lookup",
    description="Useful for looking up information about the code hierarchy codebase.",
)

"""
There is also a helpful description of the tool here, which works best as a system prompt.
"""
logger.info("There is also a helpful description of the tool here, which works best as a system prompt.")

display(Markdown("Description: " + query_engine.get_tool_instructions()))

"""
Now lets finally actually make an agent!

Note that this requires some complex reasoning, and works best with GPT-4-like LLMs.
"""
logger.info("Now lets finally actually make an agent!")


llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.1)

agent = FunctionAgent(
    tools=[tool],
    llm=llm,
    system_prompt=query_engine.get_tool_instructions(),
)


ctx = Context(agent)

async def async_func_15():
    response = await agent.run(
        "How does the get_code_hierarchy_from_nodes function from the code hierarchy node parser work? Provide specific implementation details.",
        ctx=ctx,
    )
    return response
response = asyncio.run(async_func_15())
logger.success(format_json(response))
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)