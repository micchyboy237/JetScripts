import os
import shutil
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType
from jet.vectors.semantic_search.vector_search_simple import VectorSearch

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Real-world demonstration
if __name__ == "__main__":
    # 1. Specify preffered dimensions
    dimensions = None
    # dimensions = 512
    # model_name: EmbedModelType = "mxbai-embed-large"
    # model_name: EmbedModelType = "nomic-embed-text"
    # model_name: EmbedModelType = "all-MiniLM-L6-v2"
    model_name: EmbedModelType = "embeddinggemma"
    # model_name: EmbedModelType = "static-retrieval-mrl-en-v1"
    # Same example queries
    queries = [
        "robotics use case",
    ]

    search_engine = VectorSearch(model_name, truncate_dim=dimensions)

    # Same sample documents
    sample_docs = [
        "Developers can now use Toolbox from their preferred IDE, such as Cursor, Windsurf, Claude Desktop, more and leverage our new pre-built tools such as execute\\_sql and list\\_tables for AI-assisted development with Cloud SQL for PostgreSQL, AlloyDB and self-managed PostgreSQL.",
        "+ Get Started with [MCP Toolbox for Databases](https://googleapis.github.io/genai-toolbox/getting-started/mcp_quickstart/)\n\n### Apr 28 - May 2\n\n* **Itching to build AI agents?",
        "Join the Agent Development Kit Hackathon with Google Cloud!",
        "** Use ADK to build multi-agent systems to solve challenges around complex processes, customer engagement, content creation, and more.",
        "Compete for over $50,000 in prizes and demonstrate the power of multi-agent systems with ADK and Google Cloud.",
        "+ Submissions are open from May 12, 2025 to June 23, 2025.",
        "+ Learn more and register [here](http://googlecloudmultiagents.devpost.com/).",
        "### Apr 21 - Apr 25\n\n* **Iceland's Magic: Reliving Solo Adventure through Gemini**Embark on a journey through Iceland's stunning landscapes, as experienced on Gauti's Icelandic solo trip.",
        "From majestic waterfalls to the enchanting Northern Lights, Gautami then takes these cherished memories a step further, using Google's multi-modal AI, specifically Veo2, to bring static photos to life.",
        "Discover how technology can enhance and dynamically relive travel experiences, turning precious moments into immersive short videos.",
        "This innovative approach showcases the power of AI in preserving and enriching our memories from Gauti's unforgettable Icelandic travels.",
        "[Read more](https://medium.com/@gautami_nadkarni_cloud/icelands-magic-reliving-my-solo-adventure-through-gemini-ai-d61470b9945c).",
        "* **Introducing ETLC - A Context-First Approach to Data Processing in the Generative AI Era:** As organizations adopt generative AI, data pipelines often lack the dynamic context needed.",
        "This paper introduces ETLC (Extract, Transform, Load, Contextualize), adding semantic, relational, operational, environmental, and behavioral context.",
        "ETLC enables Dynamic Context Engines for context-aware RAG, AI co-pilots, and agentic systems.",
        "It works with standards like the Model Context Protocol (MCP) for effective context delivery, ensuring business-specific AI outputs.",
        "[Read the full paper](https://services.google.com/fh/files/blogs/etlc_full_paper.pdf).",
        "### Apr 14 - Apr 18\n\n* **What's new in Database Center**  \n  With general availability, [Database Center](https://cloud.google.com/database-center/docs/overview) now provides enhanced performance and health monitoring for all Google Cloud databases, including Cloud SQL, AlloyDB, Spanner, Bigtable, Memorystore, and Firestore.",
        "It delivers richer metrics and actionable recommendations, helps you to optimize database performance and reliability, and customize your experience.",
        "Database Center also leverages Gemini to deliver assistive performance troubleshooting experience.",
        "Finally, you can track the weekly progress of your database inventory and health issues.",
        "Get started with Database Center today\n\n  + [Access Database Center in Google Cloud console](https://console.cloud.google.com/database-center)\n  + [Review the documentation to learn more](https://cloud.google.com/database-center/docs/overview)\n\n### Apr 7 - Apr 11\n\n* This week, at Google Cloud Next, we announced an expansion of Bigtable's SQL capabilities and introduced continuous materialized views.",
        "Bigtable SQL and continuous materialized views empower users to build fully-managed, real-time application backends using familiar SQL syntax, including specialized features that preserve Bigtable's flexible schema -- a vital aspect of real-time applications.",
        "Read more in this [blog](https://cloud.google.com/blog/products/databases/accelerate-your-analytics-with-new-bigtable-sql-capabilities).",
        "* **DORA Report Goes Global: Now Available in 9 Languages!",
        "**Unlock the power of DevOps insights with the DORA report, now available in 9 languages, including Chinese, French, Japanese, Korean, Portuguese, and Spanish.",
        "Global teams can now optimize their practices, benchmark performance, and gain localized insights to accelerate software delivery.",
        "The report highlights the significant impact of AI on software development, explores platform engineering's promises and challenges, and emphasizes user-centricity and stable priorities for organizational success.",
        "[Download the DORA Report Now](https://cloud.google.com/devops/state-of-devops)\n* **New Google Cloud State of AI Infrastructure Report Released**Is your infrastructure ready for AI?",
        "The 2025 State of AI Infrastructure Report is here, packed with insights from 500+ global tech leaders.",
        "Discover the strategies and challenges shaping the future of AI and learn how to build a robust, secure, and cost-effective AI-ready cloud.",
        "Download the report and enhance your AI investments today."
    ]
    search_engine.add_documents(sample_docs)

    for query in queries:
        results = search_engine.search(query, top_k=len(sample_docs))
        print(f"\nQuery: {query}")
        print("Top matches:")
        for num, (doc, score) in enumerate(results, 1):
            print(f"\n{colorize_log(f"{num}.", "ORANGE")} (Score: {
                  colorize_log(f"{score:.3f}", "SUCCESS")})")
            print(f"{doc}")

    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")
