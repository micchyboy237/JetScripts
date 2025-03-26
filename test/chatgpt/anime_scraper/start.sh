# Change to the directory where this script is located
# cd "$(dirname "$0")"

export PYTHONPATH="$PYTHONPATH:/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper"

# pip install scrapy beautifulsoup4 faiss-cpu openai requests rank-bm25 nltk

# scrapy runspider scraper/scrape_mal.py
# scrapy runspider scraper/scrape_mal_details.py
# scrapy runspider scraper/scrape_anilist.py
# python rag.py

python scraper/scrape_mal.py
python scraper/scrape_mal_details.py
# python rag.py
