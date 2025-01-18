from typing import Optional
from jet.llm.query.retrievers import setup_index
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.vectors.utils import get_source_node_attributes
from llama_index.core.readers.file.base import SimpleDirectoryReader
from pydantic.main import BaseModel


chunk_size = 1024
chunk_overlap = 20
data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
with_heirarchy = False

documents = SimpleDirectoryReader(data_dir).load_data()
query_nodes = setup_index(
    documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap, with_heirarchy=with_heirarchy)


class Metadata(BaseModel):
    chunk_size: Optional[int] = None
    depth: Optional[int] = None
    file_path: str
    file_name: str
    file_type: str
    file_size: int
    creation_date: str
    last_modified_date: str


class Node(BaseModel):
    id: str
    score: float
    text_length: int
    start_end: list[int]
    text: str
    metadata: Metadata


class NodesResponse(BaseModel):
    data: list[Node]

    @classmethod
    def from_nodes(cls, nodes: list):
        # Transform the nodes, changing 'node_id' to 'id'
        transformed_nodes = [
            {
                "id": get_source_node_attributes(node).pop("node_id"),
                **get_source_node_attributes(node),
            }
            for node in nodes
        ]
        return cls(data=transformed_nodes)


if __name__ == "__main__":
    top_k = 20
    score_threshold = 0.0
    query = "Tell me about yourself."

    # query = "Tell me about yourself and your greatest achievements."

    result = query_nodes(query, threshold=score_threshold,
                         top_k=top_k)

    data = NodesResponse.from_nodes(result["nodes"])

    logger.newline()
    logger.info("First item metadata:")
    logger.success(format_json(result["nodes"][0].metadata))

    display_jet_source_nodes(query, result["nodes"])
