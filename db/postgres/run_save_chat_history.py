import uuid
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
import psycopg

DEFAULT_DB = "chat_history_db1"
DEFAULT_USER = "jethroestrada"
DEFAULT_PASSWORD = ""
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5432

# Establish a synchronous connection to the database
conn_info = (
    f"dbname={DEFAULT_DB} user={DEFAULT_USER} password={DEFAULT_PASSWORD} "
    f"host={DEFAULT_HOST} port={DEFAULT_PORT}"
)
sync_connection = psycopg.connect(conn_info)

# Create the table schema (only needs to be done once)
table_name = "chat_history"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)

session_id = str(uuid.uuid4())

# Initialize the chat history manager
chat_history = PostgresChatMessageHistory(
    table_name,
    session_id,
    sync_connection=sync_connection
)

# Add messages to the chat history
chat_history.add_messages([
    SystemMessage(content="Meow"),
    AIMessage(content="woof"),
    HumanMessage(content="bark"),
])

print(chat_history.messages)
