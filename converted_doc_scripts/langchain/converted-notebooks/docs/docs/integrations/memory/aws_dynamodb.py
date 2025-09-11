from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.chat_message_histories import (
DynamoDBChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import boto3
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
# AWS DynamoDB

>[Amazon AWS DynamoDB](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/dynamodb/index.html) is a fully managed `NoSQL` database service that provides fast and predictable performance with seamless scalability.

This notebook goes over how to use `DynamoDB` to store chat message history with `DynamoDBChatMessageHistory` class.

## Setup

First make sure you have correctly configured the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html). Then make sure you have installed the `langchain-community` package, so we need to install that. We also need to install the `boto3` package.

```bash
pip install -U langchain-community boto3
```

It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability
"""
logger.info("# AWS DynamoDB")




"""
## Create Table

Now, create the `DynamoDB` Table where we will be storing messages:
"""
logger.info("## Create Table")


dynamodb = boto3.resource("dynamodb")

table = dynamodb.create_table(
    TableName="SessionTable",
    KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
    AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
    BillingMode="PAY_PER_REQUEST",
)

table.meta.client.get_waiter("table_exists").wait(TableName="SessionTable")

logger.debug(table.item_count)

"""
## DynamoDBChatMessageHistory
"""
logger.info("## DynamoDBChatMessageHistory")

history = DynamoDBChatMessageHistory(table_name="SessionTable", session_id="0")

history.add_user_message("hi!")

history.add_ai_message("whats up?")

history.messages

"""
## DynamoDBChatMessageHistory with Custom Endpoint URL

Sometimes it is useful to specify the URL to the AWS endpoint to connect to. For instance, when you are running locally against [Localstack](https://localstack.cloud/). For those cases you can specify the URL via the `endpoint_url` parameter in the constructor.
"""
logger.info("## DynamoDBChatMessageHistory with Custom Endpoint URL")

history = DynamoDBChatMessageHistory(
    table_name="SessionTable",
    session_id="0",
    endpoint_url="http://localhost.localstack.cloud:4566",
)

"""
## DynamoDBChatMessageHistory With Composite Keys
The default key for DynamoDBChatMessageHistory is ```{"SessionId": self.session_id}```, but you can modify this to match your table design.

### Primary Key Name
You may modify the primary key by passing in a primary_key_name value in the constructor, resulting in the following:
```{self.primary_key_name: self.session_id}```

### Composite Keys
When using an existing DynamoDB table, you may need to modify the key structure from the default of to something including a Sort Key. To do this you may use the ```key``` parameter.

Passing a value for key will override the primary_key parameter, and the resulting key structure will be the passed value.
"""
logger.info("## DynamoDBChatMessageHistory With Composite Keys")

composite_table = dynamodb.create_table(
    TableName="CompositeTable",
    KeySchema=[
        {"AttributeName": "PK", "KeyType": "HASH"},
        {"AttributeName": "SK", "KeyType": "RANGE"},
    ],
    AttributeDefinitions=[
        {"AttributeName": "PK", "AttributeType": "S"},
        {"AttributeName": "SK", "AttributeType": "S"},
    ],
    BillingMode="PAY_PER_REQUEST",
)

composite_table.meta.client.get_waiter("table_exists").wait(TableName="CompositeTable")

logger.debug(composite_table.item_count)

my_key = {
    "PK": "session_id::0",
    "SK": "langchain_history",
}

composite_key_history = DynamoDBChatMessageHistory(
    table_name="CompositeTable",
    session_id="0",
    endpoint_url="http://localhost.localstack.cloud:4566",
    key=my_key,
)

composite_key_history.add_user_message("hello, composite dynamodb table!")

composite_key_history.messages

"""
## Chaining

We can easily combine this message history class with [LCEL Runnables](/docs/how_to/message_history)

To do this we will want to use Ollama, so we need to install that
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatOllama(model="llama3.2")

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: DynamoDBChatMessageHistory(
        table_name="SessionTable", session_id=session_id
    ),
    input_messages_key="question",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "<SESSION_ID>"}}

chain_with_history.invoke({"question": "Hi! I'm bob"}, config=config)

chain_with_history.invoke({"question": "Whats my name"}, config=config)

logger.info("\n\n[DONE]", bright=True)