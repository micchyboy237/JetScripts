from flask import request
from flask import Flask, request
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from multiprocessing.managers import BaseManager
from multiprocessing import Lock
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
import os
from flask import Flask
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()


app = Flask(__name__)


@app.route("/")
def home():
    return "Hello World!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)


# os.environ["OPENAI_API_KEY"] = "your key here"

index = None


def initialize_index():
    global index
    storage_context = StorageContext.from_defaults()
    index_dir = "./.index"
    if os.path.exists(index_dir):
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader("./documents").load_data()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        storage_context.persist(index_dir)


@app.route("/query", methods=["GET"])
def query_index():
    global index
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return str(response), 200


# os.environ["OPENAI_API_KEY"] = "your key here"
index = None
lock = Lock()


def initialize_index():
    global index

    with lock:
        pass


def query_index(query_text):
    global index
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return str(response)


if __name__ == "__main__":
    print("initializing index...")
    initialize_index()

    manager = BaseManager(("", 5602), b"password")
    manager.register("query_index", query_index)
    server = manager.get_server()

    print("starting server...")
    server.serve_forever()


manager = BaseManager(("", 5602), b"password")
manager.register("query_index")
manager.connect()


@app.route("/query", methods=["GET"])
def query_index():
    global index
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    response = manager.query_index(query_text)._getvalue()
    return str(response), 200


@app.route("/")
def home():
    return "Hello World!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)

...
manager.register("insert_into_index")
...


@app.route("/uploadFile", methods=["POST"])
def upload_file():
    global manager
    if "file" not in request.files:
        return "Please send a POST request with a file", 400

    filepath = None
    try:
        uploaded_file = request.files["file"]
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join("documents", os.path.basename(filename))
        uploaded_file.save(filepath)

        if request.form.get("filename_as_doc_id", None) is not None:
            manager.insert_into_index(filepath, doc_id=filename)
        else:
            manager.insert_into_index(filepath)
    except Exception as e:
        if filepath is not None and os.path.exists(filepath):
            os.remove(filepath)
        return "Error: {}".format(str(e)), 500

    if filepath is not None and os.path.exists(filepath):
        os.remove(filepath)

    return "File inserted!", 200


def insert_into_index(doc_text, doc_id=None):
    global index
    document = SimpleDirectoryReader(input_files=[doc_text]).load_data()[0]
    if doc_id is not None:
        document.doc_id = doc_id

    with lock:
        index.insert(document)
        index.storage_context.persist()


...
manager.register("insert_into_index", insert_into_index)
...

export type Document = {
    id: string
    text: string
}

const fetchDocuments = async (): Promise < Document[] > = > {
    const response = await fetch("http://localhost:5601/getDocuments", {
        mode: "cors",
    })

    if (!response.ok) {
        return []
    }

    const documentList = (await response.json()) as Document[]
    return documentList
}

export type ResponseSources = {
    text: string
    doc_id: string
    start: number
    end: number
    similarity: number
}

export type QueryResponse = {
    text: string
    sources: ResponseSources[]
}

const queryIndex = async (query: string): Promise < QueryResponse > = > {
    const queryURL = new URL("http://localhost:5601/query?text=1")
    queryURL.searchParams.append("text", query)

    const response = await fetch(queryURL, {mode: "cors"})
    if (!response.ok) {
        return {text: "Error in query", sources: []}
    }

    const queryResponse = (await response.json()) as QueryResponse

    return queryResponse
}

export default queryIndex

const insertDocument = async (file: File) = > {
    const formData = new FormData()
    formData.append("file", file)
    formData.append("filename_as_doc_id", "true")

    const response = await fetch("http://localhost:5601/uploadFile", {
        mode: "cors",
        method: "POST",
        body: formData,
    })

    const responseText = response.text()
    return responseText
}

export default insertDocument

logger.info("\n\n[DONE]", bright=True)
