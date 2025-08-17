from jet.logger import CustomLogger
import os
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Advanced Retrieval
icon: "magnifying-glass"
iconType: "solid"
description: "Advanced memory search with keyword expansion, intelligent reranking, and precision filtering"
---

## What is Advanced Retrieval?

Advanced Retrieval gives you precise control over how memories are found and ranked. While basic search uses semantic similarity, these advanced options help you find exactly what you need, when you need it.

<CodeGroup>

## Search Enhancement Options

### Keyword Search
**Expands results** to include memories with specific terms, names, and technical keywords.

<Tabs>
  <Tab title="When to Use">
- Searching for specific entities, names, or technical terms
- Need comprehensive coverage of a topic  
- Want broader recall even if some results are less relevant
- Working with domain-specific terminology
</Tab>
<Tab title="How it Works">
"""
logger.info("## What is Advanced Retrieval?")

results = client.search(
    query="What foods should I avoid?",
    keyword_search=True,
    user_id="user123"
)

"""
</Tab>
  <Tab title="Performance">
- **Latency**: ~10ms additional
- **Recall**: Significantly increased
- **Precision**: Slightly decreased
- **Best for**: Entity search, comprehensive coverage
</Tab>
</Tabs>

### Reranking  
**Reorders results** using deep semantic understanding to put the most relevant memories first.

<Tabs>
  <Tab title="When to Use">
- Need the most relevant result at the top
- Result order is critical for your application
- Want consistent quality across different queries
- Building user-facing features where accuracy matters
</Tab>
<Tab title="How it Works">
"""
logger.info("### Reranking")

results = client.search(
    query="What are my upcoming travel plans?",
    rerank=True,
    user_id="user123"
)

"""
</Tab>
<Tab title="Performance">
- **Latency**: 150-200ms additional
- **Accuracy**: Significantly improved
- **Ordering**: Much more relevant
- **Best for**: Top-N precision, user-facing results
</Tab>
</Tabs>

### Memory Filtering
**Filters results** to keep only the most precisely relevant memories.

<Tabs>
<Tab title="When to Use">
- Need highly specific, focused results
- Working with large datasets where noise is problematic  
- Quality over quantity is essential
- Building production or safety-critical applications
</Tab>
<Tab title="How it Works">
"""
logger.info("### Memory Filtering")

results = client.search(
    query="What are my dietary restrictions?",
    filter_memories=True,
    user_id="user123"
)

"""
</Tab>
<Tab title="Performance">
- **Latency**: 200-300ms additional
- **Precision**: Maximized
- **Recall**: May be reduced
- **Best for**: Focused queries, production systems
</Tab>
</Tabs>

## Real-World Use Cases

<Tabs>
<Tab title="Personal AI Assistant">
"""
logger.info("## Real-World Use Cases")

results = client.search(
    query="How do I like my bedroom temperature?",
    keyword_search=True,    # Find specific temperature mentions
    rerank=True,           # Get most recent preferences first
    user_id="user123"
)

"""
</Tab>
<Tab title="Customer Support">
"""

results = client.search(
    query="Problems with premium subscription billing",
    keyword_search=True,     # Find "premium", "billing", "subscription"
    filter_memories=True,    # Only billing-related issues
    user_id="customer456"
)

"""
</Tab>
<Tab title="Healthcare AI">
"""

results = client.search(
    query="Patient allergies and contraindications",
    rerank=True,            # Most important info first
    filter_memories=True,   # Only medical restrictions
    user_id="patient789"
)

"""
</Tab>
<Tab title="Learning Platform">
"""

results = client.search(
    query="Python programming progress and difficulties",
    keyword_search=True,    # Find "Python", "programming", specific concepts
    rerank=True,           # Recent progress first
    user_id="student123"
)

"""
</Tab>
</Tabs>

## Choosing the Right Combination

### Recommended Configurations

<CodeGroup>
"""
logger.info("## Choosing the Right Combination")

def quick_search(query, user_id):
    return client.search(
        query=query,
        keyword_search=True,
        user_id=user_id
    )

def standard_search(query, user_id):
    return client.search(
        query=query,
        keyword_search=True,
        rerank=True,
        user_id=user_id
    )

def precise_search(query, user_id):
    return client.search(
        query=query,
        rerank=True,
        filter_memories=True,
        user_id=user_id
    )

"""

"""

function quickSearch(query, userId) {
    return client.search(query, {
        user_id: userId,
        keyword_search: true
    })
}

function standardSearch(query, userId) {
    return client.search(query, {
        user_id: userId,
        keyword_search: true,
        rerank: true
    })
}

function preciseSearch(query, userId) {
    return client.search(query, {
        user_id: userId,
        rerank: true,
        filter_memories: true
    })
}

"""
</CodeGroup>

## Best Practices

### ✅ Do
- **Start simple** with just one enhancement and measure impact
- **Use keyword search** for entity-heavy queries (names, places, technical terms)
- **Use reranking** when the top result quality matters most
- **Use filtering** for production systems where precision is critical
- **Handle empty results** gracefully when filtering is too aggressive
- **Monitor latency** and adjust based on your application's needs

### ❌ Don't
- Enable all options by default without measuring necessity
- Use filtering for broad exploratory queries
- Ignore latency impact in real-time applications
- Forget to handle cases where filtering returns no results
- Use advanced retrieval for simple, fast lookup scenarios

## Performance Guidelines

### Latency Expectations
"""
logger.info("## Best Practices")


start_time = time.time()
results = client.search(
    query="user preferences",
    keyword_search=True,  # +10ms
    rerank=True,         # +150ms
    filter_memories=True, # +250ms
    user_id="user123"
)
latency = time.time() - start_time
logger.debug(f"Search completed in {latency:.2f}s")  # ~0.41s expected

"""
### Optimization Tips

1. **Cache frequent queries** to avoid repeated advanced processing
2. **Use session-specific search** with `run_id` to reduce search space
3. **Implement fallback logic** when filtering returns empty results
4. **Monitor and alert** on search latency patterns

---

**Ready to enhance your search?** Start with keyword search for broader coverage, add reranking for better ordering, and use filtering when precision is critical.

<Snippet file="get-help.mdx" />
"""
logger.info("### Optimization Tips")

logger.info("\n\n[DONE]", bright=True)