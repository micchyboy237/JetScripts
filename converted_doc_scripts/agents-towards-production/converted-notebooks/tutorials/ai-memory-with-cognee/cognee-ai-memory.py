from jet.transformers.formatters import format_json
from IPython.display import IFrame, HTML, display
from cognee import visualize_graph
from cognee.modules.engine.models.node_set import NodeSet
from jet.logger import logger
from pathlib import Path
import cognee
import os
import shutil
import sys
import webbrowser

async def main():
    
    
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
    ![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--ai-memory-with-cognee--cognee-ai-memory)
    
    # Building Intelligent AI Memory Systems with Cognee: A Python Development Knowledge Graph
    
    This tutorial demonstrates how to construct an intelligent knowledge system that unifies authoritative Python practices, community standards, and personal development context into a coherent AI memory layer. By connecting contributions from Python's creator Guido van Rossum with established design principles and your own development patterns, you'll create a knowledge graph that produces contextually relevant, explainable, and consistent insights for Python development.
    
    
    
    ## Understanding the Architecture
    
    Modern software development involves navigating multiple sources of truth: language documentation, community best practices, historical code examples, and personal experience. Traditional approaches treat these as separate resources, requiring manual synthesis and interpretation. This tutorial demonstrates a fundamentally different approach using knowledge graphs and retrieval-augmented generation.
    
    The system we're building transforms scattered development data into an interconnected knowledge network. Rather than searching through isolated documents, you'll query a unified graph that understands relationships between Python's design philosophy, real-world implementation patterns from its creator, and your specific development context. This enables discovering non-obvious connections, such as how a type hinting challenge you faced relates to solutions Guido implemented in mypy, or how your async patterns align with Python's core design principles.
    
    The power of this approach lies in its ability to infer implicit relationships and rules from your data. Through advanced graph algorithms and AI processing, the system identifies patterns that span across different data sources, creating a memory layer that grows more intelligent with each interaction.
    
    ## System Architecture and Data Flow
    
    ```mermaid
    graph TD
        A[Raw Data Sources] --> B[Data Ingestion]
        B --> C[Knowledge Graph Construction]
        C --> D[Graph Processing]
        D --> E[Memory Layer]
        E --> F[Intelligent Search]
        F --> G[Contextual Results]
        G --> H[Feedback Loop]
        H --> E
        
        A1[Guido's Contributions] --> B
        A2[PEP Guidelines] --> B
        A3[Zen Principles] --> B
        A4[Developer Rules] --> B
        A5[Conversations] --> B
        
        C --> C1[Entity Extraction]
        C --> C2[Relationship Mapping]
        C --> C3[Temporal Connections]
        
        D --> D1[Pattern Recognition]
        D --> D2[Rule Inference]
        D --> D3[Context Synthesis]
        
        style A fill:#f9f,stroke:#333,stroke-width:2px
        style E fill:#bbf,stroke:#333,stroke-width:2px
        style G fill:#bfb,stroke:#333,stroke-width:2px
    ```
    
    ## Core Operations in Cognee
    
    The Cognee framework provides four fundamental operations that transform raw data into intelligent knowledge systems. Each operation serves a specific purpose in the knowledge graph pipeline.
    
    The **add()** function serves as the entry point for data ingestion, accepting various formats including JSON, Markdown, and API responses. This operation normalizes and prepares data for graph construction. The **cognify()** function represents the core transformation engine, using AI to extract entities, identify relationships, and structure data into a traversable knowledge graph. Through **search()**, users interact with the graph using natural language queries or structured Cypher expressions, with the system understanding context and relationships to return relevant results. Finally, **memify()** applies advanced algorithms to infer implicit connections and rules from the data, creating a dynamic memory layer that enhances search capabilities and discovers non-obvious patterns.
    
    These operations work together to create a system that not only stores information but understands and connects it in meaningful ways.
    
    ## Data Sources and Their Roles
    
    This tutorial leverages a carefully curated set of data sources that represent different perspectives on Python development. Each source contributes unique value to the knowledge graph.
    
    The **guido_contributions.json** file contains actual pull requests and commits from Guido van Rossum's work on mypy and CPython. These provide authoritative examples of how Python's creator approaches language design and problem-solving. The **pep_style_guide.md** encodes community-accepted standards for Python code style and typing conventions, ensuring that generated insights align with established best practices. The **zen_principles.md** captures Python's philosophical foundation, grounding technical decisions in principles of simplicity, explicitness, and readability.
    
    Personal context comes from **my_developer_rules.md**, which contains project-specific conventions and constraints, and **copilot_conversations.json**, which preserves actual development conversations including questions, code snippets, and discussion topics. This combination creates a comprehensive knowledge base that spans from language philosophy to practical implementation.
    
    ## Environment Setup and Configuration
    
    Cognee operates using asynchronous functions to handle complex graph operations efficiently. The following setup ensures proper execution within a Jupyter notebook environment.
    
    ### Enabling Asynchronous Execution
    
    Configure the notebook environment to support async/await operations required by Cognee's graph processing engine.
    """
    logger.info("# Building Intelligent AI Memory Systems with Cognee: A Python Development Knowledge Graph")
    
    # %pip install cognee==0.3.3
    
    # import nest_asyncio
    # nest_asyncio.apply()
    
    """
    ## Model Configuration
    
    For optimal balance between processing speed, cost efficiency, and output quality, this tutorial uses Ollama's GPT-4o-mini model. Ensure your environment configuration file (.env) contains the following setting:
    
    ```
    LLM_MODEL="llama3.2"
    ```
    
    This model provides sufficient capability for entity extraction and relationship mapping while maintaining reasonable processing times for educational purposes.
    
    ### Verifying Cognee Installation
    
    Confirm that Cognee is properly installed and identify whether you're using a local development version or an installed package.
    """
    logger.info("## Model Configuration")
    
    
    logger.debug('Quick Cognee Import Check')
    logger.debug('=' * 30)
    logger.debug(f'Cognee location: {cognee.__file__}')
    logger.debug(f'Package directory: {os.path.dirname(cognee.__file__)}')
    
    current_dir = Path.cwd()
    cognee_path = Path(cognee.__file__)
    if current_dir in cognee_path.parents:
        logger.debug('Status: LOCAL DEVELOPMENT VERSION')
    else:
        logger.debug('Status: INSTALLED PACKAGE')
    
    """
    ### Configuring Python Path
    
    Ensure the Python interpreter can locate all necessary modules by adding the project root to the system path.
    """
    logger.info("### Configuring Python Path")
    
    notebook_dir = Path.cwd()
    if notebook_dir.name == 'notebooks':
        project_root = notebook_dir.parent
    else:
        project_root = Path.cwd()
    
    project_root_str = str(project_root.absolute())
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    logger.debug(f"Project root: {project_root_str}")
    
    """
    ### Initializing Clean State
    
    Remove any existing Cognee data to ensure the tutorial starts with a fresh knowledge graph.
    """
    logger.info("### Initializing Clean State")
    
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    
    """
    ## Building the Knowledge Graph: Guido's Contributions
    
    The knowledge graph construction begins with ingesting Guido van Rossum's development history. This dataset contains detailed pull requests and commits from his work on mypy and CPython, providing concrete examples of language design decisions and implementation patterns.
    
    The ingestion process involves two key steps. First, the add() function loads the raw JSON data and assigns it to a named node set for organizational purposes. Then, cognify() processes this data to extract entities, identify relationships, and build temporal connections between different contributions. The temporal_cognify parameter enables time-based analysis, allowing queries about the evolution of Python features over time.
    
    ### Loading and Processing Guido's Development Data
    
    Ingest the contributions data and build the initial knowledge graph with temporal awareness.
    """
    logger.info("## Building the Knowledge Graph: Guido's Contributions")
    
    
    result = await cognee.add(
            os.path.abspath("data/guido_contributions.json"),
            node_set=["guido_data"]
        )
    logger.success(format_json(result))
    await cognee.cognify(temporal_cognify=True)
    
    """
    ### Examining Initial Search Results
    
    Display the first search result to verify successful graph construction.
    """
    logger.info("### Examining Initial Search Results")
    
    results = await cognee.search("Show me commits")
    logger.success(format_json(results))
    logger.debug(results[0])
    
    """
    ## Understanding Graph Structure Through Visualization
    
    The search operation demonstrates a fundamental difference from traditional database queries. Rather than retrieving isolated records, Cognee traverses a knowledge graph that understands relationships between commits, language features, design decisions, and their evolution over time. This enables discovery of patterns and connections that would be difficult to identify through conventional search methods.
    
    Visualization provides crucial insights into the graph's structure and the relationships Cognee has identified. The interactive graph reveals clustering patterns around different projects and time periods, showing how ideas evolved into features and how different contributions relate to each other.
    
    ### Generating Interactive Graph Visualization
    
    Create an HTML visualization of the knowledge graph structure.
    """
    logger.info("## Understanding Graph Structure Through Visualization")
    
    await visualize_graph('./guido_contributions.html')
    
    """
    ### Displaying the Knowledge Graph
    
    Render the interactive visualization within the notebook for exploration.
    """
    logger.info("### Displaying the Knowledge Graph")
    
    display(IFrame("./guido_contributions.html", width="100%", height="500"))
    
    
    html_path = Path('guido_contributions.html').resolve()
    file_url = html_path.as_uri()
    
    logger.debug(f"HTML file path: {html_path}")
    logger.debug(f"Opening: {file_url}")
    
    webbrowser.open(file_url)
    
    """
    ## Graph Analysis and Pattern Recognition
    
    The visualization reveals several important patterns in Python's development history. CPython core development shows concentrated activity around 2020, while mypy contributions focus specifically on fixtures and run classes. PEP discussions create connections to other contributors like Thomas Grainger and Adam Turner, showing the collaborative nature of language evolution. Time-based connections demonstrate how initial ideas and discussions evolved into concrete features and implementations.
    
    These visual patterns help understand not just what changes were made, but how different aspects of Python development interconnect. The graph structure makes it possible to trace the evolution of ideas from initial proposals through implementation and refinement.
    
    The interactive nature of the visualization allows for deeper exploration. You can examine specific clusters to understand focused development efforts or trace connections between seemingly unrelated contributions to discover hidden patterns in Python's evolution.
    
    ## Expanding the Knowledge Graph
    
    With the foundation established through Guido's contributions, the next phase involves integrating additional data sources to create a comprehensive knowledge system. This expansion connects authoritative examples with community standards, design philosophy, and personal development context.
    
    Each data source is assigned to a specific node set, creating logical groupings within the graph. Developer-specific data including conversations and personal rules forms one cluster, while principles and guidelines form another. This organization enables targeted searches within specific domains while maintaining connections across the entire graph.
    
    ### Ingesting Complete Data Set
    
    Add all remaining data sources and process them into the unified knowledge graph.
    """
    logger.info("## Graph Analysis and Pattern Recognition")
    
    
    await cognee.add(os.path.abspath("data/copilot_conversations.json"), node_set=["developer_data"])
    await cognee.add(os.path.abspath("data/my_developer_rules.md"), node_set=["developer_data"])
    await cognee.add(os.path.abspath("data/zen_principles.md"), node_set=["principles_data"])
    await cognee.add(os.path.abspath("data/pep_style_guide.md"), node_set=["principles_data"])
    
    await cognee.cognify(temporal_cognify=True)
    
    """
    ### Discovering Cross-Domain Connections
    
    Query the expanded graph to find connections between personal development challenges and Guido's solutions.
    """
    logger.info("### Discovering Cross-Domain Connections")
    
    results = await cognee.search(
            "What validation issues did I encounter in January 2024, and how would they be addressed in Guido's contributions?",
            query_type=cognee.SearchType.GRAPH_COMPLETION
        )
    logger.success(format_json(results))
    logger.debug(results)
    
    """
    ## Analyzing Connected Insights
    
    The search results demonstrate how Cognee connects disparate data sources to provide contextual insights. The system identifies patterns between your development challenges and Guido's historical solutions, revealing connections such as circular import issues in type hints that mirror problems Guido solved in specific mypy pull requests, or performance optimizations in list comprehensions that follow patterns established in CPython commits.
    
    These connections go beyond simple keyword matching. The graph understands the semantic relationships between different concepts, allowing it to recognize that a type hinting challenge you encountered relates conceptually to work done in mypy, even if the specific terminology differs.
    
    ## Advanced Memory Layer Construction
    
    The memify operation represents the most sophisticated aspect of Cognee's knowledge processing. This function applies advanced algorithms to the existing graph, inferring implicit rules and patterns that span across different data sources. Unlike basic graph construction, memify creates a dynamic memory layer that understands not just what information exists, but how different pieces of information relate to and influence each other.
    
    The memory layer enables the system to recognize patterns such as recurring design decisions in Guido's contributions that align with specific Zen principles, or common resolution strategies for particular types of Python challenges. This creates a form of institutional memory that can provide guidance based on historical patterns and established best practices.
    
    ### Building the Intelligent Memory Layer
    
    Apply advanced pattern recognition and rule inference algorithms to create the memory layer.
    """
    logger.info("## Analyzing Connected Insights")
    
    await cognee.memify()
    
    """
    ## Memory Layer Capabilities
    
    The memify operation enhances the knowledge graph with several sophisticated capabilities specific to Python development. It infers coding patterns from examples, recognizing when certain approaches consistently appear in similar contexts. The system connects abstract design philosophy to concrete implementation decisions, linking principles like "explicit is better than implicit" to specific coding choices in type hinting or API design.
    
    This creates a feedback loop where the system learns from both authoritative sources and practical experience, continuously refining its understanding of effective Python development patterns.
    
    ### Analyzing Design Pattern Alignment
    
    Query the memory-enhanced graph to understand how personal implementations align with Python philosophy.
    """
    logger.info("## Memory Layer Capabilities")
    
    results = await cognee.search(
            query_text= "How does my AsyncWebScraper implementation align with Python's design principles?",
            query_type=cognee.SearchType.GRAPH_COMPLETION
        )
    logger.success(format_json(results))
    logger.debug("Python Pattern Analysis:", results)
    
    """
    ## Targeted Search with Node Set Filtering
    
    The node set organization established during data ingestion enables precise, targeted searches within specific domains of the knowledge graph. This filtering capability proves particularly valuable when seeking authoritative guidance on specific topics, as it allows queries to focus on relevant data sources while maintaining awareness of the broader context.
    
    By constraining searches to particular node sets, you can ensure that responses draw from appropriate sources. Questions about style guidelines can be directed specifically to PEP documents and design principles, while implementation questions can focus on actual code examples and developer experiences.
    
    ### Searching Within Specific Knowledge Domains
    
    Demonstrate targeted search by querying only the principles and guidelines node set.
    """
    logger.info("## Targeted Search with Node Set Filtering")
    
    results = await cognee.search(
            query_text= "How should variables be named?",
            query_type=cognee.SearchType.GRAPH_COMPLETION,
            node_type=NodeSet,
            node_name=['principles_data']
        )
    logger.success(format_json(results))
    
    """
    ## Temporal Analysis Capabilities
    
    The temporal cognify option enabled during graph construction provides powerful capabilities for understanding how Python development has evolved over time. This temporal awareness allows queries that explore trends, identify periods of intense development activity, or understand how specific features emerged and matured.
    
    Temporal queries can reveal insights about development velocity, the relationship between different features introduced in similar timeframes, or how implementation approaches have changed as the language has evolved. This historical perspective provides valuable context for understanding current best practices and anticipating future directions.
    
    ### Exploring Time-Based Development Patterns
    
    Query the temporal aspects of the knowledge graph to understand development evolution.
    """
    logger.info("## Temporal Analysis Capabilities")
    
    await cognee.search(
        query_text = "What can we learn from Guido's contributions in 2025?",
        query_type=cognee.SearchType.TEMPORAL
    )
    
    """
    ## Implementing Continuous Learning Through Feedback
    
    The knowledge graph system supports continuous improvement through a feedback mechanism that captures the utility and relevance of search results. This creates a learning system that adapts to your specific needs and preferences over time.
    
    When search interactions are saved, the system can track which types of queries are most common, which results prove most useful, and how your development focus evolves. This feedback becomes part of the graph itself, influencing future searches and helping the system provide increasingly relevant results.
    
    ### Executing Search with Feedback Tracking
    
    Perform a search with interaction saving enabled to support future feedback.
    """
    logger.info("## Implementing Continuous Learning Through Feedback")
    
    answer = await cognee.search(
            query_type=cognee.SearchType.GRAPH_COMPLETION,
            query_text="What is the most zen thing about Python?",
            save_interaction=True,  # This enables feedback later
        )
    logger.success(format_json(answer))
    
    """
    ### Providing Feedback to Enhance Future Searches
    
    Submit feedback about the search result to improve the system's understanding of your preferences.
    """
    logger.info("### Providing Feedback to Enhance Future Searches")
    
    feedback = await cognee.search(
            query_type=cognee.SearchType.FEEDBACK,
            query_text="Last result was useful, I like code that complies with best practices.",
            last_k=1,
        )
    logger.success(format_json(feedback))
    
    """
    ## Conclusion and Next Steps
    
    This tutorial has demonstrated the construction of an intelligent knowledge system that unifies multiple perspectives on Python development into a coherent, queryable graph. The system you've built connects authoritative examples from Python's creator with community standards, design philosophy, and personal development experience, creating a powerful tool for informed decision-making in Python development.
    
    The knowledge graph approach offers significant advantages over traditional documentation and search methods. By understanding relationships between different pieces of information, the system can provide insights that would be difficult or impossible to discover through conventional means. The temporal awareness, pattern recognition, and feedback mechanisms create a dynamic system that grows more valuable over time.
    
    As you continue to use and expand this system, consider adding additional data sources such as project documentation, code reviews, or technical discussions. Each new data source enriches the graph, creating more connections and enabling more sophisticated insights. The feedback loop ensures that the system adapts to your specific needs, becoming an increasingly valuable partner in your Python development journey.
    """
    logger.info("## Conclusion and Next Steps")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())