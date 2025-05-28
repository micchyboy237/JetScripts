# Write any query about anime "Ex. Top isekai anime today" that will scrape web search engine for anime titles.
# Create tool that will accept the anime titles generated.
# Call LLM generation with this query and context for the prompt and tool.
# Will call this url. Replace placeholder with anime title.
# https://aniwatchtv.to/search?keyword=<encoded_anime_title>
# Use get_md_header_docs to scrape html header documents
# Apply search_docs with query = anime title and documents from scraped html
# Validate result with high threshold. If high score, anime exists in this site available for watching.
