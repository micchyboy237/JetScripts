from bs4 import BeautifulSoup as Soup
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
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
## Tool Use with Ollama API for structured outputs

Ollama API recently added tool use.

This is very useful for structured output.
"""
logger.info("## Tool Use with Ollama API for structured outputs")

# ! pip install -U langchain-anthropic


"""
`How can we use tools to produce structured output?`

Function call / tool use just generates a payload.

Payload often a JSON string, which can be pass to an API or, in this case, a parser to produce structured output.

LangChain has `llm.with_structured_output(schema)` to make it very easy to produce structured output that matches `schema`.

![Screenshot 2024-04-03 at 10.16.57 PM.png](attachment:83c97bfe-b9b2-48ef-95cf-06faeebaa048.png)
"""
logger.info("Function call / tool use just generates a payload.")



class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


llm = ChatOllama(
    model="claude-3-opus-20240229",
    default_headers={"anthropic-beta": "tools-2024-04-04"},
)

structured_llm = llm.with_structured_output(code, include_raw=True)
code_output = structured_llm.invoke(
    "Write a python program that prints the string 'hello world' and tell me how it works in a sentence"
)

code_output["raw"].content[0]

code_output["raw"].content[1]

code_output["raw"].content[1]["input"]

error = code_output["parsing_error"]
error

parsed_result = code_output["parsed"]

parsed_result.prefix

parsed_result.imports

parsed_result.code

"""
## More challenging example

Motivating example for tool use / structured outputs.

![code-gen.png](attachment:bb6c7126-7667-433f-ba50-56107b0341bd.png)

Here are some docs that we want to answer code questions about.
"""
logger.info("## More challenging example")


url = "https://python.langchain.com/docs/expression_language/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)

"""
Problem:

`What if we want to enforce tool use?`

We can use fallbacks.

Let's select a code gen prompt that -- from some of my testing -- does not correctly invoke the tool.

We can see if we can correct from this.
"""
logger.info("Problem:")

code_gen_prompt_working = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """<instructions> You are a coding assistant with expertise in LCEL, LangChain expression language. \n
    Here is the LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user  question based on the \n
    above provided documentation. Ensure any code you provide can be executed with all required imports and variables \n
    defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block. \n
    Invoke the code tool to structure the output correctly. </instructions> \n Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

code_gen_prompt_bad = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user
    question based on the above provided documentation. Ensure any code you provide can be executed \n
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)


class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description = "Schema for code solutions to questions about LCEL."


llm = ChatOllama(
    model="claude-3-opus-20240229",
    default_headers={"anthropic-beta": "tools-2024-04-04"},
)

structured_llm = llm.with_structured_output(code, include_raw=True)


def check_claude_output(tool_output):
    """Check for parse error or failure to call the tool"""

    if tool_output["parsing_error"]:
        logger.debug("Parsing error!")
        raw_output = str(code_output["raw"].content)
        error = tool_output["parsing_error"]
        raise ValueError(
            f"Error parsing your output! Be sure to invoke the tool. Output: {raw_output}. \n Parse error: {error}"
        )

    elif not tool_output["parsed"]:
        logger.debug("Failed to invoke tool!")
        raise ValueError(
            "You did not use the provided tool! Be sure to invoke the tool to structure the output."
        )
    return tool_output


code_chain = code_gen_prompt_bad | structured_llm | check_claude_output

"""
Let's add a check and re-try.
"""
logger.info("Let's add a check and re-try.")

def insert_errors(inputs):
    """Insert errors in the messages"""

    error = inputs["error"]
    messages = inputs["messages"]
    messages += [
        (
            "user",
            f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
        )
    ]
    return {
        "messages": messages,
        "context": inputs["context"],
    }


fallback_chain = insert_errors | code_chain
N = 3  # Max re-tries
code_chain_re_try = code_chain.with_fallbacks(
    fallbacks=[fallback_chain] * N, exception_key="error"
)

messages = [("user", "How do I build a RAG chain in LCEL?")]
code_output_lcel = code_chain_re_try.invoke(
    {"context": concatenated_content, "messages": messages}
)

parsed_result_lcel = code_output_lcel["parsed"]

parsed_result_lcel.prefix

parsed_result_lcel.imports

parsed_result_lcel.code

"""
Example trace catching an error and correcting:

https://smith.langchain.com/public/f06e62cb-2fac-46ae-80cd-0470b3155eae/r
"""
logger.info("Example trace catching an error and correcting:")


logger.info("\n\n[DONE]", bright=True)