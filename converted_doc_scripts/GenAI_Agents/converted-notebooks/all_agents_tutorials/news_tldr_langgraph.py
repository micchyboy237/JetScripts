async def main():
    from jet.transformers.formatters import format_json
    from IPython.display import display, Image as IPImage
    from bs4 import BeautifulSoup
    from datetime import datetime
    from dotenv import load_dotenv
    from jet.llm.ollama.base_langchain import ChatOllama
    from jet.logger import CustomLogger
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables.graph import MermaidDrawMethod
    from langgraph.graph import Graph, END
    from newsapi import NewsApiClient
    from pydantic import BaseModel, Field
    from typing import TypedDict, Annotated, List
    import asyncio
    import os
    import re
    import requests
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # News TL;DR using Langgraph (Too Long Didn't Read)
    
    ## Overview
    This project demonstrates the creation of a news summarization agent uses large language models (LLMs) for decision making and summarization as well as a news API calls. The integration of LangGraph to coordinate sequential and cyclical processes, open-ai to choose and condense articles, newsAPI to retrieve relevant article metadata, and BeautifulSoup for web scraping allows for the generation of relevant current event article TL;DRs from a single query.
    
    ## Motivation
    Although LLMs demonstrate excellent conversational and educational ability, they lack access to knowledge of current events. This project allow users to ask about a news topic they are interested and receive a TL;DR of relevant articles. The goal is to allow users to conveniently follow their interest and stay current with their connection to world events.
    
    ## Key Components
    1. **LangGraph**: Orchestrates the overall workflow, managing the flow of data between different stages of the process.
    2. **GPT-4o-mini (via LangChain)**: Generates search terms, selects relevant articles, parses html, provides article summaries
    3. **NewsAPI**: Retrieves article metadata from keyword search
    4. **BeautifulSoup**: Retrieves html from page
    5. **Asyncio**: Allows separate LLM calls to be made concurrently for speed efficiency.
    
    ## Method
    The news research follows these high-level steps:
    
    1. **NewsAPI Parameter Creation (LLM 1)**: Given a user query, the model generates a formatted parameter dict for the news search.
    
    2. **Article Metadata Retrieval**: An API call to NewsAPI retrieves relevant article metadata.
    
    3. **Article Text Retrieval**: Beautiful Soup scrapes the full article text from the urls to ensure validity.
    
    4. **Conditional Logic**: Conditional logic either: repeats 1-3 if article threshold not reached, proceeds to step 5, end with no articles found.
    
    5. **Relevant Article Selection (LLM 2)**: The model selects urls from the most relevant n-articles for the user query based on the short synopsis provided by the API.
    
    6. **Generate TL;DR (LLM 3+)**: A summarized set of bullet points for each article is generated concurrently with Asyncio.
    
    This workflow is managed by LangGraph to make sure that the appropriate prompt is fed to the each LLM call.
    
    ## Conclusion
    This news TL;DR agent highlights the utility of coordinating successive LLM generations in order to
    achieve a higher level goal.
    
    Although the current implementation only retrieves bulleted summaries, it could be elaborated to start
    a dialogue with the user that could allow them to ask questions about the article and get 
    more information or to collectively generate a coherent opinion.
    
    ## Setup and Imports
    
    Install and import necessary libraries
    """
    logger.info("# News TL;DR using Langgraph (Too Long Didn't Read)")
    
    # !pip install langgraph -q
    # !pip install langchain-ollama -q
    # !pip install langchain-core -q
    # !pip install pydantic -q
    # !pip install python-dotenv -q
    # !pip install newsapi-python -q
    # !pip install beautifulsoup4 -q
    # !pip install ipython -q
    # !pip install nest_asyncio -q
    
    
    # from getpass import getpass
    
    
    
    """
    ## Get an NewsAPI Key
    * create a free developer account at https://newsapi.org/
    * 100 requests per day
    * articles between 1 day and 1 month old
    
    ## Setup LLM Model
    * create an account and register a credit card at https://platform.ollama.com/chat-completions
    * create an API key
    
    ## Create Your Environmental Variables (Optional)
    Create a file named `.env` in the same directory as this notebook with the following
    ```
    # OPENAI_API_KEY = 'your-api-key'
    NEWSAPI_KEY = 'your-api-key'
    ```
    
    If you skip this step, you will be asked to input all API keys once each time you start this notebook.
    
    ## Initialize Model and Environmental Variables
    
    # If you're not running a local model with Ollama, the next cell will ask for your OPENAI_API_KEY and
    securely add it as an environmental variable. It will not persist in this notebook.
    """
    logger.info("## Get an NewsAPI Key")
    
    if os.path.exists("../.env"):
        load_dotenv()
    else:
    #     os.environ["NEWSAPI_KEY"] = getpass("Enter your News API key: ")
    #     os.environ["OPENAI_API_KEY"] = getpass("Enter your Ollama API key: ")
    
    model = "llama3.2"
    llm = ChatOllama(model="llama3.2")
    
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key:
            logger.debug("NEWSAPI_KEY successfully loaded from .env.")
    
    """
    ## Test APIs
    """
    logger.info("## Test APIs")
    
    llm.invoke("Why is the sky blue?").content
    
    newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY'))
    
    query = 'ai news of the day'
    
    all_articles = newsapi.get_everything(q=query,
                                          sources='google-news,bbc-news,techcrunch',
                                          domains='techcrunch.com, bbc.co.uk',
                                          language='en',
                                          sort_by='relevancy',)
    
    
    all_articles['articles'][0]
    
    """
    ## Define Data Structures
    
    Define the GraphState class. Each user query will be added to a new instance of this class, which will be passed
    through the LangGraph structure while collect outputs from each step. When it reaches the END node, it's final
    result will be returned to the user.
    """
    logger.info("## Define Data Structures")
    
    class GraphState(TypedDict):
        news_query: Annotated[str, "Input query to extract news search parameters from."]
        num_searches_remaining: Annotated[int, "Number of articles to search for."]
        newsapi_params: Annotated[dict, "Structured argument for the News API."]
        past_searches: Annotated[List[dict], "List of search params already used."]
        articles_metadata: Annotated[list[dict], "Article metadata response from the News API"]
        scraped_urls: Annotated[List[str], "List of urls already scraped."]
        num_articles_tldr: Annotated[int, "Number of articles to create TL;DR for."]
        potential_articles: Annotated[List[dict[str, str, str]], "Article with full text to consider summarizing."]
        tldr_articles: Annotated[List[dict[str, str, str]], "Selected article TL;DRs."]
        formatted_results: Annotated[str, "Formatted results to display."]
    
    """
    ## Define NewsAPI argument data structure with Pydantic
    * the model will create a formatted dictionary of params for the NewsAPI call
    * the NewsApiParams class inherits from the Pydantic BaseModel
    * Langchain will parse and feed paramd descriptions to the LLM
    """
    logger.info("## Define NewsAPI argument data structure with Pydantic")
    
    class NewsApiParams(BaseModel):
        q: str = Field(description="1-3 concise keyword search terms that are not too specific")
        sources: str =Field(description="comma-separated list of sources from: 'abc-news,abc-news-au,associated-press,australian-financial-review,axios,bbc-news,bbc-sport,bloomberg,business-insider,cbc-news,cbs-news,cnn,financial-post,fortune'")
        from_param: str = Field(description="date in format 'YYYY-MM-DD' Two days ago minimum. Extend up to 30 days on second and subsequent requests.")
        to: str = Field(description="date in format 'YYYY-MM-DD' today's date unless specified")
        language: str = Field(description="language of articles 'en' unless specified one of ['ar', 'de', 'en', 'es', 'fr', 'he', 'it', 'nl', 'no', 'pt', 'ru', 'se', 'ud', 'zh']")
        sort_by: str = Field(description="sort by 'relevancy', 'popularity', or 'publishedAt'")
    
    """
    ## Define Graph Functions
    
    Define the functions (nodes) that will be used in the LangGraph workflow.
    """
    logger.info("## Define Graph Functions")
    
    def generate_newsapi_params(state: GraphState) -> GraphState:
        """Based on the query, generate News API params."""
        parser = JsonOutputParser(pydantic_object=NewsApiParams)
    
        today_date = datetime.now().strftime("%Y-%m-%d")
    
        past_searches = state["past_searches"]
    
        num_searches_remaining = state["num_searches_remaining"]
    
        news_query = state["news_query"]
    
        template = """
        Today is {today_date}.
    
        Create a param dict for the News API based on the user query:
        {query}
    
        These searches have already been made. Loosen the search terms to get more results.
        {past_searches}
    
        Following these formatting instructions:
        {format_instructions}
    
        Including this one, you have {num_searches_remaining} searches remaining.
        If this is your last search, use all news sources and a 30 days search range.
        """
    
        prompt_template = PromptTemplate(
            template=template,
            variables={"today": today_date, "query": news_query, "past_searches": past_searches, "num_searches_remaining": num_searches_remaining},
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
    
        chain = prompt_template | llm | parser
    
        result = chain.invoke({"query": news_query, "today_date": today_date, "past_searches": past_searches, "num_searches_remaining": num_searches_remaining})
    
        state["newsapi_params"] = result
    
        return state
    
    def retrieve_articles_metadata(state: GraphState) -> GraphState:
        """Using the NewsAPI params, perform api call."""
        newsapi_params = state["newsapi_params"]
    
        state['num_searches_remaining'] -= 1
    
        try:
            newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY'))
    
            articles = newsapi.get_everything(**newsapi_params)
    
            state['past_searches'].append(newsapi_params)
    
            scraped_urls = state["scraped_urls"]
    
            new_articles = []
            for article in articles['articles']:
                if article['url'] not in scraped_urls and len(state['potential_articles']) + len(new_articles) < 10:
                    new_articles.append(article)
    
            state["articles_metadata"] = new_articles
    
        except Exception as e:
            logger.debug(f"Error: {e}")
    
        return state
    
    def retrieve_articles_text(state: GraphState) -> GraphState:
        """Web scrapes to retrieve article text."""
        articles_metadata = state["articles_metadata"]
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
        }
    
        potential_articles = []
    
        for article in articles_metadata:
            url = article['url']
    
            response = requests.get(url, headers=headers)
    
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
    
                text = soup.get_text(strip=True)
    
                potential_articles.append({"title": article["title"], "url": url, "description": article["description"], "text": text})
    
                state["scraped_urls"].append(url)
    
        state["potential_articles"].extend(potential_articles)
    
        return state
    
    def select_top_urls(state: GraphState) -> GraphState:
        """Based on the article synoses, choose the top-n articles to summarize."""
        news_query = state["news_query"]
        num_articles_tldr = state["num_articles_tldr"]
    
        potential_articles = state["potential_articles"]
    
        formatted_metadata = "\n".join([f"{article['url']}\n{article['description']}\n" for article in potential_articles])
    
        prompt = f"""
        Based on the user news query:
        {news_query}
    
        Reply with a list of strings of up to {num_articles_tldr} relevant urls.
        Don't add any urls that are not relevant or aren't listed specifically.
        {formatted_metadata}
        """
        result = llm.invoke(prompt).content
    
        url_pattern = r'(https?://[^\s",]+)'
    
        urls = re.findall(url_pattern, result)
    
        tldr_articles = [article for article in potential_articles if article['url'] in urls]
    
        state["tldr_articles"] = tldr_articles
    
        return state
    
    async def summarize_articles_parallel(state: GraphState) -> GraphState:
        """Summarize the articles based on full text."""
        tldr_articles = state["tldr_articles"]
    
    
        prompt = """
        Create a * bulleted summarizing tldr for the article:
        {text}
    
        Be sure to follow the following format exaxtly with nothing else:
        {title}
        {url}
        * tl;dr bulleted summary
        * use bullet points for each sentence
        """
    
        for i in range(len(tldr_articles)):
            text = tldr_articles[i]["text"]
            title = tldr_articles[i]["title"]
            url = tldr_articles[i]["url"]
            result = llm.invoke(prompt.format(title=title, url=url, text=text))
            tldr_articles[i]["summary"] = result.content
    
        state["tldr_articles"] = tldr_articles
    
        return state
    
    def format_results(state: GraphState) -> GraphState:
        """Format the results for display."""
        q = [newsapi_params["q"] for newsapi_params in state["past_searches"]]
        formatted_results = f"Here are the top {len(state['tldr_articles'])} articles based on search terms:\n{', '.join(q)}\n\n"
    
        tldr_articles = state["tldr_articles"]
    
        tldr_articles = "\n\n".join([f"{article['summary']}" for article in tldr_articles])
    
        formatted_results += tldr_articles
    
        state["formatted_results"] = formatted_results
    
        return state
    
    """
    ## Set Up LangGraph Workflow
    
    Set up decision logic to try to retrieve `num_searches_remaining` articles, while limiting attempts to 5.
    """
    logger.info("## Set Up LangGraph Workflow")
    
    def articles_text_decision(state: GraphState) -> str:
        """Check results of retrieve_articles_text to determine next step."""
        if state["num_searches_remaining"] == 0:
            if len(state["potential_articles"]) == 0:
                state["formatted_results"] = "No articles with text found."
                return "END"
            else:
                return "select_top_urls"
        else:
            if len(state["potential_articles"]) < state["num_articles_tldr"]:
                return "generate_newsapi_params"
            else:
                return "select_top_urls"
    
    """
    Define the LangGraph workflow by adding nodes and edges.
    """
    logger.info("Define the LangGraph workflow by adding nodes and edges.")
    
    workflow = Graph()
    
    workflow.set_entry_point("generate_newsapi_params")
    
    workflow.add_node("generate_newsapi_params", generate_newsapi_params)
    workflow.add_node("retrieve_articles_metadata", retrieve_articles_metadata)
    workflow.add_node("retrieve_articles_text", retrieve_articles_text)
    workflow.add_node("select_top_urls", select_top_urls)
    workflow.add_node("summarize_articles_parallel", summarize_articles_parallel)
    workflow.add_node("format_results", format_results)
    
    workflow.add_edge("generate_newsapi_params", "retrieve_articles_metadata")
    workflow.add_edge("retrieve_articles_metadata", "retrieve_articles_text")
    workflow.add_conditional_edges(
        "retrieve_articles_text",
        articles_text_decision,
        {
            "generate_newsapi_params": "generate_newsapi_params",
            "select_top_urls": "select_top_urls",
            "END": END
        }
        )
    workflow.add_edge("select_top_urls", "summarize_articles_parallel")
    workflow.add_conditional_edges(
        "summarize_articles_parallel",
        lambda state: "format_results" if len(state["tldr_articles"]) > 0 else "END",
        {
            "format_results": "format_results",
            "END": END
        }
        )
    workflow.add_edge("format_results", END)
    
    app = workflow.compile()
    
    """
    ## Display Graph Structure
    """
    logger.info("## Display Graph Structure")
    
    display(
        IPImage(
            app.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )
    
    """
    ## Run Workflow Function
    
    Define a function to run the workflow and display results.
    """
    logger.info("## Run Workflow Function")
    
    async def run_workflow(query: str, num_searches_remaining: int = 10, num_articles_tldr: int = 3):
        """Run the LangGraph workflow and display results."""
        initial_state = {
            "news_query": query,
            "num_searches_remaining": num_searches_remaining,
            "newsapi_params": {},
            "past_searches": [],
            "articles_metadata": [],
            "scraped_urls": [],
            "num_articles_tldr": num_articles_tldr,
            "potential_articles": [],
            "tldr_articles": [],
            "formatted_results": "No articles with text found."
        }
        try:
            result = await app.ainvoke(initial_state)
            logger.success(format_json(result))
    
            return result["formatted_results"]
        except Exception as e:
            logger.debug(f"An error occurred: {str(e)}")
            return None
    
    """
    ## Execute Workflow
    
    Run the workflow with a sample query.
    """
    logger.info("## Execute Workflow")
    
    query = "what are the top genai news of today?"
    logger.debug(await run_workflow(query, num_articles_tldr=3))
    
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