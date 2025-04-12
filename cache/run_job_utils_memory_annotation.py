

from jet.cache.joblib.memory import memory


@memory(output_dir)
def generate_header_docs(html: str):
    return get_docs_from_html(html)


@memory(output_dir)
def generate_header_tokens(header_docs, embed_model):
    return get_header_tokens_and_update_metadata(header_docs, embed_model)


@memory(output_dir)
def generate_all_header_nodes(header_docs, header_tokens, query, llm_model, embed_models, sub_chunk_size, sub_chunk_overlap):
    return get_all_header_nodes(header_docs, header_tokens, query, llm_model, embed_models, sub_chunk_size, sub_chunk_overlap)


@memory(output_dir)
def generate_reranked_nodes(query, filtered_nodes, embed_models, header_map):
    return rerank_nodes(query, filtered_nodes, embed_models, header_map)


@memory(output_dir)
def generate_grouped_nodes(nodes, llm_model, max_tokens):
    return group_nodes(nodes, llm_model, max_tokens)


@memory(output_dir)
def generate_context_schema(query, context, llm_model):
    return generate_browser_query_context_json_schema(query, context, llm_model)


@memory(output_dir)
def get_structured_response(output_cls, llm, query, context):
    return llm.structured_predict(
        output_cls,
        prompt=SEARCH_WEB_PROMPT_TEMPLATE,
        model=llm.model,
        context=context,
        instruction=SYSTEM_QUERY_SCHEMA_DOCS,
        query=query,
    )
