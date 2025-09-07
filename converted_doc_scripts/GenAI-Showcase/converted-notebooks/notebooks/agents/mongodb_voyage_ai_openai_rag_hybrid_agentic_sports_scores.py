async def main():
    from jet.transformers.formatters import format_json
    from agents import Agent, Runner
    from agents.tool import function_tool
    from datetime import datetime, timedelta
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from ollama import Ollama
    from pymongo import MongoClient
    import logging
    import os
    import shutil
    import voyageai
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # MongoDB Atlas Vector Search with VoyageAI Embeddings for Sports Scores and Stories
    
    This notebook demonstrates how to use VoyageAI embeddings with MongoDB Atlas Vector Search for retrieving relevant sports scores and stories based on user queries.
    
    ## Overview
    
    In this tutorial, we'll learn how to:
    
    1. Connect to MongoDB Atlas and retrieve sports data
    2. Generate embeddings using VoyageAI's embedding models
    3. Store these embeddings in MongoDB
    4. Create and use a vector search index for semantic similarity search
    5. Use hybrid search for result tuning.
    6. Implement a RAG (Retrieval-Augmented Generation) system to answer questions about sports teams and matches
    7. Showing how Agentic rag changes the results by using hybrid search as tools for an ai-agent built with the ollama-agent sdk.
    
    This approach combines the power of vector embeddings with natural language processing to provide relevant sports information based on user queries.
    
    ## Setup and Configuration
    
    First, let's import the necessary libraries and set up our environment. We'll need libraries for data manipulation, machine learning, visualization, and MongoDB connectivity.
    """
    logger.info("# MongoDB Atlas Vector Search with VoyageAI Embeddings for Sports Scores and Stories")
    
    # %pip install voyageai pymongo   scikit-learn python-dotenv ollama
    
    
    
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    load_dotenv()
    
    """
    ### Environment Variables
    
    We'll use environment variables to store sensitive information like API keys and connection strings. These should be stored in a `.env` file in the same directory as this notebook.
    
    Example `.env` file content:
    ```
    MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
    VOYAGE_API_KEY=your_voyage_api_key_here
    # OPENAI_API_KEY=your_ollama_api_key_here
    ```
    """
    logger.info("### Environment Variables")
    
    # import getpass
    
    # MONGODB_URI = getpass.getpass("Enter your MongoDB connection string: ")
    # VOYAGE_API_KEY = getpass.getpass("Enter your VoyageAI API key: ")
    # OPENAI_API_KEY = getpass.getpass("Enter your Ollama API key: ")
    
    
    # if not MONGODB_URI or not VOYAGE_API_KEY or not OPENAI_API_KEY:
        logger.debug(
    #         "Error: Environment variables MONGODB_URI, VOYAGE_API_KEY, and OPENAI_API_KEY must be set"
        )
        logger.debug("Please create a .env file with these variables")
    else:
        logger.debug("Environment variables loaded successfully")
    
    """
    ### MongoDB Configuration
    
    Now let's set up our MongoDB connection and define the database and collections we'll be using.
    """
    logger.info("### MongoDB Configuration")
    
    DB_NAME = "sports_demo"
    COLLECTION_NAME = "matches"
    TEAMS_COLLECTION = "teams"
    NEWS_COLLECTION = "news"
    VECTOR_COLLECTION = "vector_features"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "voyage_vector_index"
    
    client = MongoClient(MONGODB_URI, appname="voyageai.mongodb.sports_scores_demo")
    
    matches_collection = client[DB_NAME][COLLECTION_NAME]
    teams_collection = client[DB_NAME][TEAMS_COLLECTION]
    news_collection = client[DB_NAME][NEWS_COLLECTION]
    vector_collection = client[DB_NAME][VECTOR_COLLECTION]
    
    try:
        client.admin.command("ismaster")
        logger.debug("MongoDB connection successful")
    except Exception as e:
        logger.debug(f"MongoDB connection failed: {e}")
    
    """
    ## VoyageAI Embeddings
    
    Next, we'll create a class to handle generating embeddings using VoyageAI's API. Embeddings are vector representations of text that capture semantic meaning, allowing us to perform operations like similarity search.
    """
    logger.info("## VoyageAI Embeddings")
    
    class VoyageAIEmbeddings:
        """Custom VoyageAI embeddings class"""
    
        def __init__(self, api_key, model="voyage-3"):
            self.api_key = api_key
            self.model = model
            os.environ["VOYAGE_API_KEY"] = api_key
            self.client = voyageai.Client(api_key=api_key)
    
        def embed_text(self, text):
            """Embed a single text using VoyageAI"""
            response = self.client.embed([text], model=self.model, input_type="document")
            return response.embeddings[0]
    
        def embed_batch(self, texts, batch_size=20):
            """Embed a batch of texts efficiently"""
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self.client.embed(batch, model=self.model, input_type="document")
                embeddings.extend(response.embeddings)
                logger.debug(f"Processed {i+len(batch)}/{len(texts)} embeddings")
            return embeddings
    
    """
    ### Understanding Embeddings
    
    Embeddings are dense vector representations of text that capture semantic meaning. The VoyageAI model we're using (`voyage-3`) generates 1024-dimensional vectors for each text input. These vectors have several important properties:
    
    1. **Semantic similarity**: Texts with similar meanings will have embeddings that are close to each other in the vector space
    2. **Dimensionality**: The high-dimensional space allows for capturing complex relationships between concepts
    3. **Language understanding**: The model has been trained on vast amounts of text data to understand language nuances
    
    In our case, we'll use these embeddings to represent sports data in a way that captures the semantic meaning of team names, match descriptions, and news stories.
    
    ## Sample Data Generation
    
    For demonstration purposes, let's create some sample sports data. In a real-world scenario, this data would come from an API or another data source.
    """
    logger.info("### Understanding Embeddings")
    
    def generate_sample_data():
        """Generate sample sports data for demonstration purposes"""
        logger.debug("Generating sample sports data...")
    
        teams = [
            {
                "team_id": "MNU",
                "name": "Manchester United",
                "nicknames": ["Red Devils", "United"],
                "league": "Premier League",
                "country": "England",
            },
            {
                "team_id": "MNC",
                "name": "Manchester City",
                "nicknames": ["Citizens", "City"],
                "league": "Premier League",
                "country": "England",
            },
            {
                "team_id": "LIV",
                "name": "Liverpool",
                "nicknames": ["Reds", "The Kop"],
                "league": "Premier League",
                "country": "England",
            },
            {
                "team_id": "CHE",
                "name": "Chelsea",
                "nicknames": ["Blues", "The Pensioners"],
                "league": "Premier League",
                "country": "England",
            },
            {
                "team_id": "ARS",
                "name": "Arsenal",
                "nicknames": ["Gunners", "The Arsenal"],
                "league": "Premier League",
                "country": "England",
            },
            {
                "team_id": "TOT",
                "name": "Tottenham Hotspur",
                "nicknames": ["Spurs", "Lilywhites"],
                "league": "Premier League",
                "country": "England",
            },
            {
                "team_id": "BAR",
                "name": "Barcelona",
                "nicknames": ["Barça", "Blaugrana"],
                "league": "La Liga",
                "country": "Spain",
            },
            {
                "team_id": "RMA",
                "name": "Real Madrid",
                "nicknames": ["Los Blancos", "Merengues"],
                "league": "La Liga",
                "country": "Spain",
            },
            {
                "team_id": "ATM",
                "name": "Atletico Madrid",
                "nicknames": ["Atleti", "Colchoneros"],
                "league": "La Liga",
                "country": "Spain",
            },
            {
                "team_id": "BAY",
                "name": "Bayern Munich",
                "nicknames": ["Die Roten", "Bavarians"],
                "league": "Bundesliga",
                "country": "Germany",
            },
            {
                "team_id": "BVB",
                "name": "Borussia Dortmund",
                "nicknames": ["BVB", "Die Schwarzgelben"],
                "league": "Bundesliga",
                "country": "Germany",
            },
            {
                "team_id": "JUV",
                "name": "Juventus",
                "nicknames": ["Old Lady", "Bianconeri"],
                "league": "Serie A",
                "country": "Italy",
            },
            {
                "team_id": "INT",
                "name": "Inter Milan",
                "nicknames": ["Nerazzurri", "La Beneamata"],
                "league": "Serie A",
                "country": "Italy",
            },
            {
                "team_id": "ACM",
                "name": "AC Milan",
                "nicknames": ["Rossoneri", "Diavolo"],
                "league": "Serie A",
                "country": "Italy",
            },
            {
                "team_id": "PSG",
                "name": "Paris Saint-Germain",
                "nicknames": ["Les Parisiens", "PSG"],
                "league": "Ligue 1",
                "country": "France",
            },
        ]
    
        now = datetime.now()
        matches = []
    
        matches.extend(
            [
                {
                    "match_id": "PL2023-001",
                    "home_team": "MNU",
                    "away_team": "LIV",
                    "home_score": 2,
                    "away_score": 1,
                    "date": (now - timedelta(days=2)).strftime("%Y-%m-%d"),
                    "competition": "Premier League",
                    "season": "2023-2024",
                    "stadium": "Old Trafford",
                    "summary": "Manchester United secured a thrilling 2-1 victory over Liverpool at Old Trafford. Bruno Fernandes opened the scoring with a penalty in the 34th minute, before Marcus Rashford doubled the lead with a brilliant solo effort in the 67th minute. Mohamed Salah pulled one back for Liverpool in the 85th minute, but United held on for a crucial win.",
                },
                {
                    "match_id": "PL2023-002",
                    "home_team": "ARS",
                    "away_team": "MNC",
                    "home_score": 1,
                    "away_score": 1,
                    "date": (now - timedelta(days=3)).strftime("%Y-%m-%d"),
                    "competition": "Premier League",
                    "season": "2023-2024",
                    "stadium": "Emirates Stadium",
                    "summary": "Arsenal and Manchester City played out an entertaining 1-1 draw at the Emirates Stadium. Erling Haaland gave City the lead in the 23rd minute with a powerful header, but Bukayo Saka equalized for the Gunners in the 59th minute with a well-placed shot from the edge of the box.",
                },
                {
                    "match_id": "PL2023-003",
                    "home_team": "CHE",
                    "away_team": "TOT",
                    "home_score": 3,
                    "away_score": 0,
                    "date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "competition": "Premier League",
                    "season": "2023-2024",
                    "stadium": "Stamford Bridge",
                    "summary": "Chelsea dominated Tottenham in a 3-0 London derby win at Stamford Bridge. Cole Palmer scored twice in the first half, and Nicolas Jackson added a third in the 78th minute to complete the rout. Spurs struggled to create chances throughout the match.",
                },
            ]
        )
    
        matches.extend(
            [
                {
                    "match_id": "LL2023-001",
                    "home_team": "BAR",
                    "away_team": "RMA",
                    "home_score": 3,
                    "away_score": 2,
                    "date": (now - timedelta(days=4)).strftime("%Y-%m-%d"),
                    "competition": "La Liga",
                    "season": "2023-2024",
                    "stadium": "Camp Nou",
                    "summary": "Barcelona edged Real Madrid 3-2 in an exciting El Clásico at Camp Nou. Robert Lewandowski scored twice for Barça, while Lamine Yamal added another. Vinícius Júnior and Jude Bellingham scored for Real Madrid, but it wasn't enough to prevent defeat.",
                },
                {
                    "match_id": "LL2023-002",
                    "home_team": "ATM",
                    "away_team": "BAR",
                    "home_score": 1,
                    "away_score": 2,
                    "date": (now - timedelta(days=11)).strftime("%Y-%m-%d"),
                    "competition": "La Liga",
                    "season": "2023-2024",
                    "stadium": "Metropolitano",
                    "summary": "Barcelona came from behind to beat Atletico Madrid 2-1 at the Metropolitano. Antoine Griezmann gave Atletico the lead in the first half, but goals from Pedri and Robert Lewandowski in the second half secured the win for Barcelona.",
                },
            ]
        )
    
        matches.extend(
            [
                {
                    "match_id": "BL2023-001",
                    "home_team": "BAY",
                    "away_team": "BVB",
                    "home_score": 4,
                    "away_score": 0,
                    "date": (now - timedelta(days=5)).strftime("%Y-%m-%d"),
                    "competition": "Bundesliga",
                    "season": "2023-2024",
                    "stadium": "Allianz Arena",
                    "summary": "Bayern Munich thrashed Borussia Dortmund 4-0 in Der Klassiker at the Allianz Arena. Harry Kane scored a hat-trick, while Leroy Sané added another as Bayern dominated from start to finish.",
                },
                {
                    "match_id": "SA2023-001",
                    "home_team": "JUV",
                    "away_team": "INT",
                    "home_score": 1,
                    "away_score": 1,
                    "date": (now - timedelta(days=6)).strftime("%Y-%m-%d"),
                    "competition": "Serie A",
                    "season": "2023-2024",
                    "stadium": "Allianz Stadium",
                    "summary": "Juventus and Inter Milan shared the points in a 1-1 draw in the Derby d'Italia. Dusan Vlahovic put Juventus ahead in the first half, but Lautaro Martínez equalized for Inter in the second half.",
                },
            ]
        )
    
        news = [
            {
                "news_id": "NEWS001",
                "title": "Manchester United's Bruno Fernandes wins Player of the Month",
                "date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                "content": "Manchester United captain Bruno Fernandes has been named Premier League Player of the Month for his outstanding performances. The Portuguese midfielder scored 4 goals and provided 3 assists in 5 matches, helping United climb up the table. This is Fernandes' 5th Player of the Month award since joining United in January 2020.",
                "teams": ["MNU"],
                "players": ["Bruno Fernandes"],
                "category": "Award",
            },
            {
                "news_id": "NEWS002",
                "title": "Liverpool suffer injury blow as Salah ruled out for three weeks",
                "date": now.strftime("%Y-%m-%d"),
                "content": "Liverpool have been dealt a major injury blow with the news that Mohamed Salah will be sidelined for three weeks with a hamstring strain. The Egyptian forward picked up the injury during Liverpool's 2-1 defeat to Manchester United and is expected to miss crucial matches against Arsenal and Manchester City. Manager Jürgen Klopp described the injury as 'unfortunate timing' as Liverpool enter a busy period of fixtures.",
                "teams": ["LIV", "MNU"],
                "players": ["Mohamed Salah"],
                "category": "Injury",
            },
            {
                "news_id": "NEWS003",
                "title": "Barcelona's Lamine Yamal becomes youngest El Clásico goalscorer",
                "date": (now - timedelta(days=4)).strftime("%Y-%m-%d"),
                "content": "Barcelona wonderkid Lamine Yamal has made history by becoming the youngest ever goalscorer in El Clásico at just 16 years and 107 days old. The Spanish teenager scored a spectacular long-range goal in Barcelona's 3-2 victory over Real Madrid at Camp Nou. 'It's a dream come true,' said Yamal after the match. 'I've been watching El Clásico since I was a child, and to score in this fixture is incredible.'",
                "teams": ["BAR", "RMA"],
                "players": ["Lamine Yamal"],
                "category": "Record",
            },
            {
                "news_id": "NEWS004",
                "title": "Manchester City's Erling Haaland on track to break Premier League scoring record",
                "date": (now - timedelta(days=2)).strftime("%Y-%m-%d"),
                "content": "Manchester City striker Erling Haaland is on course to break his own Premier League scoring record this season. The Norwegian has already netted 15 goals in just 10 matches, putting him ahead of his record-breaking pace from last season when he scored 36 goals. Pep Guardiola praised Haaland's incredible form: 'What he's doing is remarkable. His hunger for goals is insatiable.'",
                "teams": ["MNC"],
                "players": ["Erling Haaland"],
                "category": "Performance",
            },
            {
                "news_id": "NEWS005",
                "title": "Bayern Munich's Harry Kane scores perfect hat-trick in Der Klassiker",
                "date": (now - timedelta(days=5)).strftime("%Y-%m-%d"),
                "content": "Harry Kane scored a perfect hat-trick (right foot, left foot, header) as Bayern Munich demolished Borussia Dortmund 4-0 in Der Klassiker. The England captain has made a sensational start to his Bundesliga career since his summer move from Tottenham Hotspur. 'I'm loving my time here in Munich,' said Kane. 'The team is incredible and we're playing some fantastic football.'",
                "teams": ["BAY", "BVB"],
                "players": ["Harry Kane"],
                "category": "Performance",
            },
        ]
    
        teams_collection.delete_many({})
        matches_collection.delete_many({})
        news_collection.delete_many({})
    
        teams_collection.insert_many(teams)
        matches_collection.insert_many(matches)
        news_collection.insert_many(news)
    
        logger.debug(
            f"Inserted {len(teams)} teams, {len(matches)} matches, and {len(news)} news stories"
        )
    
        return teams, matches, news
    
    
    teams, matches, news = generate_sample_data()
    
    """
    ## Data Processing and Embedding Generation
    
    Now let's define functions to process our sports data and generate embeddings.
    """
    logger.info("## Data Processing and Embedding Generation")
    
    def generate_text_for_embedding(item, item_type):
        """Create a text representation for embedding based on the item type"""
        if item_type == "match":
            home_team = next(
                (team["name"] for team in teams if team["team_id"] == item["home_team"]),
                item["home_team"],
            )
            away_team = next(
                (team["name"] for team in teams if team["team_id"] == item["away_team"]),
                item["away_team"],
            )
    
            text_parts = [
                f"Match: {home_team} vs {away_team}",
                f"Score: {item['home_score']}-{item['away_score']}",
                f"Competition: {item['competition']} {item['season']}",
                f"Date: {item['date']}",
                f"Stadium: {item['stadium']}",
                f"Summary: {item['summary']}",
            ]
            return " ".join(text_parts)
    
        elif item_type == "team":
            text_parts = [
                f"Team: {item['name']}",
                f"Also known as: {', '.join(item['nicknames'])}",
                f"League: {item['league']}",
                f"Country: {item['country']}",
            ]
            return " ".join(text_parts)
    
        elif item_type == "news":
            text_parts = [
                f"Title: {item['title']}",
                f"Date: {item['date']}",
                f"Category: {item['category']}",
                f"Content: {item['content']}",
            ]
            return " ".join(text_parts)
    
        return ""
    
    
    def create_and_save_embeddings():
        """Generate and save embeddings for all sports data"""
        logger.debug("Generating embeddings for sports data...")
    
        voyage_embeddings = VoyageAIEmbeddings(api_key=VOYAGE_API_KEY)
    
        vector_collection.delete_many({})
    
        team_texts = [generate_text_for_embedding(team, "team") for team in teams]
        team_embeddings = voyage_embeddings.embed_batch(team_texts)
    
        match_texts = [generate_text_for_embedding(match, "match") for match in matches]
        match_embeddings = voyage_embeddings.embed_batch(match_texts)
    
        news_texts = [generate_text_for_embedding(news_item, "news") for news_item in news]
        news_embeddings = voyage_embeddings.embed_batch(news_texts)
    
        vector_records = []
    
        for i, team in enumerate(teams):
            vector_records.append(
                {
                    "object_id": team["team_id"],
                    "object_type": "team",
                    "name": team["name"],
                    "league": team["league"],
                    "country": team["country"],
                    "embedding": team_embeddings[i],
                    "data": team,
                }
            )
    
        for i, match in enumerate(matches):
            vector_records.append(
                {
                    "object_id": match["match_id"],
                    "object_type": "match",
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "competition": match["competition"],
                    "date": match["date"],
                    "embedding": match_embeddings[i],
                    "data": match,
                }
            )
    
        for i, news_item in enumerate(news):
            vector_records.append(
                {
                    "object_id": news_item["news_id"],
                    "object_type": "news",
                    "title": news_item["title"],
                    "date": news_item["date"],
                    "category": news_item["category"],
                    "embedding": news_embeddings[i],
                    "data": news_item,
                }
            )
    
        vector_collection.insert_many(vector_records)
        logger.debug(f"Saved {len(vector_records)} embedding records to MongoDB")
    
        return vector_records
    
    def create_vector_search_index():
        """Create a vector search index in MongoDB Atlas"""
    
        logger.debug("Setting up Vector Search Index in MongoDB Atlas...")
        logger.debug("Note: To create the vector search index in MongoDB Atlas:")
        logger.debug("1. Go to the MongoDB Atlas dashboard")
        logger.debug("2. Select your cluster")
        logger.debug("3. Go to the 'Search' tab")
        logger.debug(
            f"4. Create a new index on '{VECTOR_COLLECTION}'with the following configuration:"
        )
        logger.debug("""
       {
      "fields": [
        {
          "type": "vector",
          "path": "embedding",
          "numDimensions": 1024,
          "similarity": "cosine"
        }
      ]
    }
        """)
        logger.debug(f"Name the index: {ATLAS_VECTOR_SEARCH_INDEX_NAME}")
        logger.debug("5. Apply the index to the vector_features collection")
    
    
    def perform_vector_search(query_text, k=5):
        """Perform a vector search query using VoyageAI embeddings"""
        logger.debug(f"Performing vector search for: {query_text}")
    
        voyage_embeddings = VoyageAIEmbeddings(api_key=VOYAGE_API_KEY)
        query_embedding = voyage_embeddings.client.embed(
            [query_text], model=voyage_embeddings.model, input_type="query"
        ).embeddings[0]
    
        vector_search_results = vector_collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": k,
                    }
                },
                {
                    "$project": {
                        "object_id": 1,
                        "object_type": 1,
                        "name": 1,
                        "title": 1,
                        "competition": 1,
                        "date": 1,
                        "data": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
        )
    
        results = list(vector_search_results)
    
        logger.debug(f"Found {len(results)} relevant items:")
        for i, result in enumerate(results):
            if result["object_type"] == "team":
                logger.debug(
                    f"{i+1}. Team: {result.get('name', 'Unknown')} (Score: {result.get('score', 0):.4f})"
                )
            elif result["object_type"] == "match":
                home = result.get("data", {}).get("home_team", "Unknown")
                away = result.get("data", {}).get("away_team", "Unknown")
                score = f"{result.get('data', {}).get('home_score', 0)}-{result.get('data', {}).get('away_score', 0)}"
                logger.debug(
                    f"{i+1}. Match: {home} vs {away} ({score}) (Score: {result.get('score', 0):.4f})"
                )
            elif result["object_type"] == "news":
                logger.debug(
                    f"{i+1}. News: {result.get('title', 'Unknown')} (Score: {result.get('score', 0):.4f})"
                )
    
        return results
    
    vector_records = create_and_save_embeddings()
    
    create_vector_search_index()
    
    example_queries = [
        "Recent Manchester United games",
        "The Red Devils, how did they do?",
        "Who won El Clasico?",
        "Premier League match results",
        "Player injuries news",
        "Bayern Munich performance",
    ]
    
    logger.debug("Testing vector search with example queries:")
    for query in example_queries:
        logger.debug("\n" + "=" * 50)
        logger.debug(f"QUERY: {query}")
        logger.debug("=" * 50)
        results = perform_vector_search(query, k=10)
    
    """
    ## Hybrid Search
    
    [Hybrid Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/) allows combination of full text search for text token matching with vector search for semantic mapping.
    """
    logger.info("## Hybrid Search")
    
    def create_full_search_index():
        """Create a fulltext search index in MongoDB Atlas"""
    
        logger.debug("Setting up Search Index in MongoDB Atlas...")
        logger.debug("Note: To create the vector search index in MongoDB Atlas:")
        logger.debug("1. Go to the MongoDB Atlas dashboard")
        logger.debug("2. Select your cluster")
        logger.debug("3. Go to the 'Search' tab")
        logger.debug(
            f"4. Create a new 'Search' index on '{VECTOR_COLLECTION}'with the following configuration:"
        )
        logger.debug("""
       {
      "mappings": {
        "dynamic": true,
        }
      }
    }
        """)
        logger.debug("Name the index: default")
        logger.debug("5. Apply the index to the vector_features collection")
    
    def hybrid_search(query, limit=5, vector_weight=0.5, full_text_weight=0.5):
        """Perform a hybrid search using vector search and full-text search."""
    
        voyage_embeddings = VoyageAIEmbeddings(api_key=VOYAGE_API_KEY)
        query_embedding = voyage_embeddings.client.embed(
            [query], model=voyage_embeddings.model, input_type="query"
        ).embeddings[0]
    
        pipeline = [
            {
                "$vectorSearch": {
                    "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": limit * 2,  # Get more results for potential ranking
                }
            },
            {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
            {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
            {
                "$addFields": {
                    "vs_score": {
                        "$multiply": [
                            vector_weight,
                            {
                                "$divide": [
                                    1.0,
                                    {
                                        "$add": ["$rank", 60]  # Adjust ranking
                                    },
                                ]
                            },
                        ]
                    }
                }
            },
            {
                "$project": {
                    "vs_score": 1,
                    "_id": "$docs._id",
                    "title": "$docs.title",
                    "object_type": "$docs.object_type",
                    "data": "$docs.data",
                }
            },
            {
                "$unionWith": {
                    "coll": VECTOR_COLLECTION,
                    "pipeline": [
                        {
                            "$search": {
                                "index": "default",
                                "compound": {
                                    "must": [
                                        {
                                            "text": {
                                                "query": query,
                                                "path": {"wildcard": "*"},
                                                "fuzzy": {},
                                            }
                                        }
                                    ]
                                },
                            }
                        },
                        {"$limit": limit * 2},
                        {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                        {"$unwind": {"path": "$docs", "includeArrayIndex": "fts_rank"}},
                        {
                            "$addFields": {
                                "fts_score": {
                                    "$multiply": [
                                        full_text_weight,
                                        {"$divide": [1.0, {"$add": ["$fts_rank", 60]}]},
                                    ]
                                }
                            }
                        },
                        {
                            "$project": {
                                "fts_score": 1,
                                "_id": "$docs._id",
                                "title": "$docs.title",
                                "object_type": "$docs.object_type",
                                "data": "$docs.data",
                            }
                        },
                    ],
                }
            },
            {
                "$addFields": {
                    "final_score": {
                        "$add": [
                            {"$ifNull": ["$vs_score", 0]},  # Handle missing vs_score
                            {"$ifNull": ["$fts_score", 0]},  # Handle missing fts_score
                        ]
                    }
                }
            },
            {"$sort": {"final_score": -1}},
            {"$limit": limit},
        ]
    
        results = list(vector_collection.aggregate(pipeline))
    
        logger.debug(f"Found {len(results)} relevant items:")
        for i, result in enumerate(results):
            if result["object_type"] == "team":
                logger.debug(
                    f"{i+1}. Team: {result.get('data', {}).get('name', 'Unknown')} (Score: {result.get('final_score', 0):.4f})"
                )
            elif result["object_type"] == "match":
                home = result.get("data", {}).get("home_team", "Unknown")
                away = result.get("data", {}).get("away_team", "Unknown")
                score = f"{result.get('data', {}).get('home_score', 0)}-{result.get('data', {}).get('away_score', 0)}"
                logger.debug(
                    f"{i+1}. Match: {home} vs {away} ({score}) (Score: {result.get('final_score', 0):.4f})"
                )
            elif result["object_type"] == "news":
                logger.debug(
                    f"{i+1}. News: {result.get('data', {}).get('title', 'Unknown')} (Score: {result.get('final_score', 0):.4f})"
                )
    
        return results
    
    example_queries = [
        "Recent Manchester United games",
        "The Red Devils, how did they do?",
        "Who won El Clasico?",
        "Premier League match results",
        "Player injuries news",
        "Bayern Munich performance",
    ]
    
    logger.debug("Testing vector search with default wieghts example queries:")
    for query in example_queries:
        logger.debug("\n" + "=" * 50)
        logger.debug(f"QUERY: {query}")
        logger.debug("=" * 50)
        results = hybrid_search(query, limit=5)
    
        logger.debug("Testing vector search with favor of vector wieghts example queries:")
    for query in example_queries:
        logger.debug("\n" + "=" * 50)
        logger.debug(f"QUERY: {query}")
        logger.debug("=" * 50)
        results = hybrid_search(query, limit=5, vector_weight=0.9, full_text_weight=0.1)
    
    """
    
    
    ## RAG with Ollama
    
    RAG is a pipeline that loads similarity or hybrid context into an LLM to produce a relevant response considering a specific question.
    """
    logger.info("## RAG with Ollama")
    
    
    # client = Ollama(api_key=OPENAI_API_KEY)
    
    
    def generate_response_with_hybrid_search(query, limit=5):
        """Generates a response using Ollama's responses API with hybrid search."""
    
        search_results = hybrid_search(query, limit=limit)
    
        context = ""
        for result in search_results:
            if result["object_type"] == "team":
                context += f"Team: {result.get('data', {}).get('name', 'Unknown')}\n"
            elif result["object_type"] == "match":
                home = result.get("data", {}).get("home_team", "Unknown")
                away = result.get("data", {}).get("away_team", "Unknown")
                score = f"{result.get('data', {}).get('home_score', 0)}-{result.get('data', {}).get('away_score', 0)}"
                context += f"Match: {home} vs {away} ({score})\n"
            elif result["object_type"] == "news":
                context += f"News: {result.get('data', {}).get('title', 'Unknown')}\n{result.get('data', {}).get('content', '')}\n"
    
        response = client.chat.completions.create(
            model="llama3.2", log_dir=f"{LOG_DIR}/chats",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful sports assistant. Answer the user's query using the provided context.",
                },
                {"role": "user", "content": f"{query}\n\nContext:\n{context}"},
            ],
        )
    
        return response.choices[0].message.content
    
    
    def generate_response_with_vector_search(query, limit=5):
        """Generates a response using Ollama's responses API with vector search."""
    
        search_results = perform_vector_search(query, k=limit)
    
        context = ""
        for result in search_results:
            if result["object_type"] == "team":
                context += f"Team: {result.get('name', 'Unknown')}\n"
            elif result["object_type"] == "match":
                home = result.get("data", {}).get("home_team", "Unknown")
                away = result.get("data", {}).get("away_team", "Unknown")
                score = f"{result.get('data', {}).get('home_score', 0)}-{result.get('data', {}).get('away_score', 0)}"
                context += f"Match: {home} vs {away} ({score})\n"
            elif result["object_type"] == "news":
                context += f"News: {result.get('title', 'Unknown')}\n{result.get('data', {}).get('content', '')}\n"
    
        response = client.chat.completions.create(
            model="llama3.2", log_dir=f"{LOG_DIR}/chats",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful sports assistant. Answer the user's query using the provided context.",
                },
                {"role": "user", "content": f"{query}\n\nContext:\n{context}"},
            ],
        )
    
        return response.choices[0].message.content
    
    query = "Who won El Clasico?"
    
    logger.debug("Testing hybrid search with example queries:")
    logger.debug("=" * 50)
    response_hybrid = generate_response_with_hybrid_search(query)
    
    logger.debug("=" * 20 + "Hybrid RAG" + "=" * 20)
    logger.debug("Response (Hybrid Search):", response_hybrid)
    
    logger.debug("\nTesting vector search with example queries:")
    logger.debug("=" * 50)
    response_vector = generate_response_with_vector_search(query)
    logger.debug("=" * 20 + "Vector RAG" + "=" * 20)
    logger.debug("Response (Vector Search):", response_vector)
    
    
    """
    ## Agentic RAG with Hybrid Search
    
    Here we will use the [ollama-agents](https://ollama.github.io/ollama-agents-python/) sdk to use the "hybrid_search" function as a tool. This helps the AI to better tailor the search term we pass to the tools and can perform multiple step tasks.
    """
    logger.info("## Agentic RAG with Hybrid Search")
    
    # !pip install -Uq ollama-agents
    
    OPENAI_MODEL = "gpt-4o"
    
    
    
    @function_tool
    def hybrid_search(
        query: str, limit: int, vector_weight: float, full_text_weight: float
    ) -> list:
        """Perform a hybrid search using vector search and full-text search."""
    
        voyage_embeddings = VoyageAIEmbeddings(api_key=VOYAGE_API_KEY)
        query_embedding = voyage_embeddings.client.embed(
            [query], model=voyage_embeddings.model, input_type="query"
        ).embeddings[0]
    
        pipeline = [
            {
                "$vectorSearch": {
                    "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": limit * 2,  # Get more results for potential ranking
                }
            },
            {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
            {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
            {
                "$addFields": {
                    "vs_score": {
                        "$multiply": [
                            vector_weight,
                            {
                                "$divide": [
                                    1.0,
                                    {
                                        "$add": ["$rank", 60]  # Adjust ranking
                                    },
                                ]
                            },
                        ]
                    }
                }
            },
            {
                "$project": {
                    "vs_score": 1,
                    "_id": "$docs._id",
                    "title": "$docs.title",
                    "object_type": "$docs.object_type",
                    "data": "$docs.data",
                }
            },
            {
                "$unionWith": {
                    "coll": VECTOR_COLLECTION,
                    "pipeline": [
                        {
                            "$search": {
                                "index": "default",
                                "compound": {
                                    "must": [
                                        {
                                            "text": {
                                                "query": query,
                                                "path": {"wildcard": "*"},
                                                "fuzzy": {},
                                            }
                                        }
                                    ]
                                },
                            }
                        },
                        {"$limit": limit * 2},
                        {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                        {"$unwind": {"path": "$docs", "includeArrayIndex": "fts_rank"}},
                        {
                            "$addFields": {
                                "fts_score": {
                                    "$multiply": [
                                        full_text_weight,
                                        {"$divide": [1.0, {"$add": ["$fts_rank", 60]}]},
                                    ]
                                }
                            }
                        },
                        {
                            "$project": {
                                "fts_score": 1,
                                "_id": "$docs._id",
                                "title": "$docs.title",
                                "object_type": "$docs.object_type",
                                "data": "$docs.data",
                            }
                        },
                    ],
                }
            },
            {
                "$addFields": {
                    "final_score": {
                        "$add": [
                            {"$ifNull": ["$vs_score", 0]},  # Handle missing vs_score
                            {"$ifNull": ["$fts_score", 0]},  # Handle missing fts_score
                        ]
                    }
                }
            },
            {"$sort": {"final_score": -1}},
            {"$limit": limit},
        ]
    
        results = list(vector_collection.aggregate(pipeline))
    
        logger.debug(f"Found {len(results)} relevant items:")
        for i, result in enumerate(results):
            if result["object_type"] == "team":
                logger.debug(
                    f"{i+1}. Team: {result.get('data', {}).get('name', 'Unknown')} (Score: {result.get('final_score', 0):.4f})"
                )
            elif result["object_type"] == "match":
                home = result.get("data", {}).get("home_team", "Unknown")
                away = result.get("data", {}).get("away_team", "Unknown")
                score = f"{result.get('data', {}).get('home_score', 0)}-{result.get('data', {}).get('away_score', 0)}"
                logger.debug(
                    f"{i+1}. Match: {home} vs {away} ({score}) (Score: {result.get('final_score', 0):.4f})"
                )
            elif result["object_type"] == "news":
                logger.debug(
                    f"{i+1}. News: {result.get('data', {}).get('title', 'Unknown')} (Score: {result.get('final_score', 0):.4f})"
                )
    
        return results
    
    
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    virtual_primary_care_assistant = Agent(
        name="Sports Assistant specialised on sports queries",
        model=OPENAI_MODEL,
        instructions="""
          You can search information using the tools hybrid_search, be excited like you are a fun!
        """,
        tools=[hybrid_search],
    )
    
    example_queries = [
        "Recent Manchester United games",
        "The Red Devils, how did they do?",
        "Who won El Clasico?",
        "Premier League match results",
        "Player injuries news",
        "Bayern Munich performance",
    ]
    
    
    logger.debug("Testing agentic hybrid search with example queries:")
    logger.debug("=" * 50)
    
    for query in example_queries:
        logger.debug("\n" + "=" * 50)
        logger.debug(f"QUERY: {query}")
        logger.debug("=" * 50)
        run_result_with_tools = await Runner.run(
                virtual_primary_care_assistant, input=query
            )
        logger.success(format_json(run_result_with_tools))
        logger.debug(run_result_with_tools.final_output)
        logger.debug("=" * 50)
    
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