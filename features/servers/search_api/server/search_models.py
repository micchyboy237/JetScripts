from pydantic import BaseModel, Field
from typing import Optional, Literal, Union
from datetime import date


class SearchQueryResult(BaseModel):
    query: str = Field(..., description="The search query string")


class DirectoryResult(BaseModel):
    directory: str = Field(..., description="The output directory path")


class MessageResult(BaseModel):
    message: str = Field(...,
                         description="A message describing the step result")


class CountResult(BaseModel):
    count: int = Field(..., ge=0,
                       description="The count of items processed or fetched")


class UrlStatusResult(BaseModel):
    url: str = Field(..., description="The URL being scraped")
    status: Literal["started", "completed"] = Field(
        ..., description="The status of the URL scraping")


class TotalTokensResult(BaseModel):
    total_tokens: int = Field(..., ge=0,
                              description="The total number of tokens counted")


class ContextSegmentsResult(BaseModel):
    context_segments: int = Field(..., ge=0,
                                  description="The number of context segments built")


class ResponseContentResult(BaseModel):
    content: str = Field(...,
                         description="A chunk of the streamed response content")


class FullResponseResult(BaseModel):
    message: str = Field(...,
                         description="Message indicating response generation status")
    full_response: str = Field(...,
                               description="The complete generated response")


class ContextRelevance(BaseModel):
    relevance_score: float = Field(..., ge=0.0, le=1.0,
                                   description="Relevance score for context")
    is_valid: bool = Field(..., description="Whether the context is valid")
    error: Optional[str] = Field(
        None, description="Error message if context is invalid")


class ResponseRelevance(BaseModel):
    relevance_score: float = Field(..., ge=0.0, le=1.0,
                                   description="Relevance score for response")
    is_valid: bool = Field(..., description="Whether the response is valid")
    error: Optional[str] = Field(
        None, description="Error message if response is invalid")


class ContextInfo(BaseModel):
    model: str = Field(..., description="The language model used")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    contexts: list[dict] = Field(..., description="List of context details")
    headers_stats: dict = Field(..., description="Statistics about headers")


class FinalSearchResult(BaseModel):
    query: str = Field(..., description="The original search query")
    context: str = Field(..., description="The context used for the response")
    response: str = Field(..., description="The final response generated")
    context_info: ContextInfo = Field(...,
                                      description="Metadata about the context and model")


class StepResult(BaseModel):
    step_title: str = Field(...,
                            description="The title of the processing step")
    step_result: Union[
        SearchQueryResult,
        DirectoryResult,
        MessageResult,
        CountResult,
        UrlStatusResult,
        TotalTokensResult,
        ContextSegmentsResult,
        ResponseContentResult,
        FullResponseResult,
        ContextRelevance,
        ResponseRelevance,
        FinalSearchResult
    ] = Field(..., description="The result of the processing step")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: int = Field(
        10, ge=1, le=50, description="Number of top results to return")
    embed_model: str = Field("static-retrieval-mrl-en-v1",
                             description="Embedding model type")
    llm_model: str = Field("llama-3.2-1b-instruct-4bit",
                           description="LLM model type")
    seed: int = Field(45, description="Random seed for reproducibility")
    use_cache: bool = Field(
        False, description="Whether to use cached search results")
    min_mtld: float = Field(
        100.0, ge=0.0, description="Minimum MTLD score for filtering")
    stream: bool = Field(False, description="Whether to stream the response")


class SearchResponse(BaseModel):
    query: str
    context: str
    response: str
    context_info: dict
    headers_stats: dict


class StreamResponseChunk(BaseModel):
    step_title: str = Field(..., description="Title of the processing step")
    step_result: Optional[dict] = Field(
        None, description="Result of the processing step")
