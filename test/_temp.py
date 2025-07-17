from jet.logger.config import colorize_log
from jet.vectors.semantic_search.vector_search_simple import VectorSearch


if __name__ == "__main__":
    # Real-world demonstration
    search_engine = VectorSearch("mxbai-embed-large")

    chunk = "Bilue is a digital consultancy that designs and builds smart, user-friendly technology for some of Australia's most well-known businesses.\nFrom mobile apps to beautifully designed web platforms and digital experiences, we create solutions that drive impact and deliver exceptional customer outcomes.\nOur culture is people-first and purpose-driven.\nWe're a down-to-earth, values-led team with offices in Sydney and Melbourne, and a growing presence in Manila.\nWe genuinely enjoy working together, whether we're solving tough tech problems, brainstorming creative solutions, or grabbing a coffee between meetings.\nCuriosity is encouraged.\nCollaboration is second nature.".lower()
    # Same sample documents
    sample_docs = chunk.splitlines()

    search_engine.add_documents(sample_docs)

    # Same example queries
    queries = [
        "react",
        "react web",
        "web",
    ]
    # Apply query template
    # QUERY_TEMPLATE = "Is this relevant to this query?\nQuery: {query}"
    # queries = [QUERY_TEMPLATE.format(query=query) for query in queries]

    for query in queries:
        results = search_engine.search(query)
        print(f"\nQuery: {query}")
        print("Top matches:")
        for doc, score in results:
            print(
                f"- {doc} (score: {colorize_log(f"{score:.3f}", color="SUCCESS")})")
