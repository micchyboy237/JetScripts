# Sample documents (similar to RAG task inputs)
import json
from visualization.plot_topics import process_documents_for_chart


documents = [
    {"id": 1, "content": "Advances in artificial intelligence are transforming industries."},
    {"id": 2, "content": "Stock market trends indicate a bullish economy."},
    {"id": 3, "content": "Machine learning models improve prediction accuracy."},
    {"id": 4, "content": "New vaccine developed for infectious disease."},
    {"id": 5, "content": "Neural networks are key to modern AI systems."},
    {"id": 6, "content": "Investment strategies for a volatile market."}
]

# Generate Chart.js configuration
chart_config = process_documents_for_chart(documents)

# Save to JSON for use in HTML
with open("chart_config.json", "w") as f:
    json.dump(chart_config, f, indent=2)

# Print for reference
print(json.dumps(chart_config, indent=2))
