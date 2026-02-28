# Complete the shell script that will create the usage example file structure. Use this prefix for each filename: "<2_digit_char>_demo_*", ex: 01_demo_*

# cd <path_to_base_dir>

#!/bin/bash

# This script creates a usage example file structure
# All examples have prefix: <2_digit>_demo_*

set -e

EXAMPLES_DIR="auto_extraction_examples"
mkdir -p "$EXAMPLES_DIR"
cd "$EXAMPLES_DIR"

# 01_demo_schema_autogen.py
cat > 01_demo_schema_autogen.py <<'EOF'
import json
import asyncio
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig, CacheMode
from crawl4ai import JsonCssExtractionStrategy

async def smart_extraction_workflow():
    """
    Step 1: Generate schema once using LLM
    Step 2: Cache schema for unlimited reuse
    Step 3: Extract from thousands of pages with zero LLM calls
    """
    cache_dir = Path("./schema_cache")
    cache_dir.mkdir(exist_ok=True)
    schema_file = cache_dir / "product_schema.json"
    if schema_file.exists():
        schema = json.load(schema_file.open())
        print("✅ Using cached schema (FREE)")
    else:
        print("🔄 Generating schema (ONE-TIME LLM COST)...")
        llm_config = LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token="env:OPENAI_API_KEY"
        )
        async with AsyncWebCrawler() as crawler:
            sample_result = await crawler.arun(
                url="https://example.com/products",
                config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            )
            sample_html = sample_result.cleaned_html[:8000]
        schema = JsonCssExtractionStrategy.generate_schema(
            html=sample_html,
            schema_type="CSS",
            query="Extract product information including name, price, description, features",
            llm_config=llm_config
        )
        json.dump(schema, schema_file.open("w"), indent=2)
        print("✅ Schema generated and cached")
    strategy = JsonCssExtractionStrategy(schema, verbose=True)
    config = CrawlerRunConfig(
        extraction_strategy=strategy,
        cache_mode=CacheMode.BYPASS
    )
    urls = [
        "https://example.com/products",
        "https://example.com/electronics",
        "https://example.com/books"
    ]
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            result = await crawler.arun(url=url, config=config)
            if result.success:
                data = json.loads(result.extracted_content)
                print(f"✅ {url}: Extracted {len(data)} items (FREE)")

asyncio.run(smart_extraction_workflow())
EOF

# 02_demo_autogen_target_json.py
cat > 02_demo_autogen_target_json.py <<'EOF'
import json
from crawl4ai import JsonCssExtractionStrategy

# When you know exactly what JSON structure you want
target_json_example = """
{
    "name": "Product Name",
    "price": "$99.99",
    "rating": 4.5,
    "features": ["feature1", "feature2"],
    "description": "Product description"
}
"""

# sample_html and llm_config are assumed to be obtained beforehand
# schema = JsonCssExtractionStrategy.generate_schema(
#     html=sample_html,
#     target_json_example=target_json_example,
#     llm_config=llm_config
# )
EOF

# 03_demo_manual_css.py
cat > 03_demo_manual_css.py <<'EOF'
import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import JsonCssExtractionStrategy

# Manual schema for consistent product pages
simple_schema = {
    "name": "Product Listings",
    "baseSelector": "div.product-card",
    "fields": [
        {"name": "title", "selector": "h2.product-title", "type": "text"},
        {"name": "price", "selector": ".price", "type": "text"},
        {"name": "image_url", "selector": "img.product-image", "type": "attribute", "attribute": "src"},
        {"name": "product_url", "selector": "a.product-link", "type": "attribute", "attribute": "href"},
        {"name": "rating", "selector": ".rating", "type": "attribute", "attribute": "data-rating"}
    ]
}

async def extract_products():
    strategy = JsonCssExtractionStrategy(simple_schema, verbose=True)
    config = CrawlerRunConfig(extraction_strategy=strategy)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/products",
            config=config
        )
        if result.success:
            products = json.loads(result.extracted_content)
            print(f"Extracted {len(products)} products")
            for product in products[:3]:
                print(f"- {product['title']}: {product['price']}")

asyncio.run(extract_products())
EOF

# 04_demo_manual_complex_nested.py
cat > 04_demo_manual_complex_nested.py <<'EOF'
import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import JsonCssExtractionStrategy

complex_schema = {
    "name": "E-commerce Product Catalog",
    "baseSelector": "div.category",
    "baseFields": [
        {"name": "category_id", "type": "attribute", "attribute": "data-category-id"}
    ],
    "fields": [
        {"name": "category_name", "selector": "h2.category-title", "type": "text"},
        {
            "name": "products",
            "selector": "div.product",
            "type": "nested_list",
            "fields": [
                {"name": "name", "selector": "h3.product-name", "type": "text"},
                {"name": "price", "selector": "span.price", "type": "text"},
                {
                    "name": "details",
                    "selector": "div.product-details",
                    "type": "nested",
                    "fields": [
                        {"name": "brand", "selector": "span.brand", "type": "text"},
                        {"name": "model", "selector": "span.model", "type": "text"}
                    ]
                },
                {
                    "name": "features",
                    "selector": "ul.features li",
                    "type": "list",
                    "fields": [{"name": "feature", "type": "text"}]
                },
                {
                    "name": "reviews",
                    "selector": "div.review",
                    "type": "nested_list",
                    "fields": [
                        {"name": "reviewer", "selector": "span.reviewer-name", "type": "text"},
                        {"name": "rating", "selector": "span.rating", "type": "attribute", "attribute": "data-rating"}
                    ]
                }
            ]
        }
    ]
}

async def extract_complex_ecommerce():
    strategy = JsonCssExtractionStrategy(complex_schema, verbose=True)
    config = CrawlerRunConfig(
        extraction_strategy=strategy,
        js_code="window.scrollTo(0, document.body.scrollHeight);",
        wait_for="css:.product:nth-child(10)"
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/complex-catalog",
            config=config
        )
        if result.success:
            data = json.loads(result.extracted_content)
            for category in data:
                print(f"Category: {category['category_name']}")
                print(f"Products: {len(category.get('products', []))}")

asyncio.run(extract_complex_ecommerce())
EOF

# 05_demo_xpath.py
cat > 05_demo_xpath.py <<'EOF'
from crawl4ai import JsonXPathExtractionStrategy

xpath_schema = {
    "name": "News Articles with XPath",
    "baseSelector": "//article[@class='news-item']",
    "fields": [
        {"name": "headline", "selector": ".//h2[contains(@class, 'headline')]", "type": "text"},
        {"name": "author", "selector": ".//span[@class='author']/text()", "type": "text"},
        {"name": "publish_date", "selector": ".//time/@datetime", "type": "text"},
        {"name": "content", "selector": ".//div[@class='article-body']//text()", "type": "text"}
    ]
}

strategy = JsonXPathExtractionStrategy(xpath_schema, verbose=True)
EOF

# 06_demo_regex_builtin.py
cat > 06_demo_regex_builtin.py <<'EOF'
import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import RegexExtractionStrategy

async def extract_common_patterns():
    strategy = RegexExtractionStrategy(
        pattern=(
            RegexExtractionStrategy.Email |
            RegexExtractionStrategy.PhoneUS |
            RegexExtractionStrategy.Url |
            RegexExtractionStrategy.Currency |
            RegexExtractionStrategy.DateIso
        )
    )
    config = CrawlerRunConfig(extraction_strategy=strategy)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/contact",
            config=config
        )
        if result.success:
            matches = json.loads(result.extracted_content)
            by_type = {}
            for match in matches:
                label = match['label']
                by_type.setdefault(label, []).append(match['value'])
            for pattern_type, values in by_type.items():
                print(f"{pattern_type}: {len(values)} matches")
                for value in values[:3]:
                    print(f"  {value}")

asyncio.run(extract_common_patterns())
EOF

# 07_demo_regex_custom.py
cat > 07_demo_regex_custom.py <<'EOF'
import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import RegexExtractionStrategy

async def extract_custom_patterns():
    custom_patterns = {
        "product_sku": r"SKU[-:]?\s*([A-Z0-9]{4,12})",
        "discount": r"(\d{1,2})%\s*off",
        "model_number": r"Model\s*#?\s*([A-Z0-9-]+)",
        "isbn": r"ISBN[-:]?\s*(\d{10}|\d{13})",
        "stock_ticker": r"\$([A-Z]{2,5})",
        "version": r"v(\d+\.\d+(?:\.\d+)?)"
    }
    strategy = RegexExtractionStrategy(custom=custom_patterns)
    config = CrawlerRunConfig(extraction_strategy=strategy)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/products",
            config=config
        )
        if result.success:
            data = json.loads(result.extracted_content)
            for item in data:
                print(f"{item['label']}: {item['value']}")

asyncio.run(extract_custom_patterns())
EOF

# 08_demo_regex_llm.py
cat > 08_demo_regex_llm.py <<'EOF'
import json
import asyncio
from pathlib import Path
from crawl4ai import AsyncWebCrawler, RegexExtractionStrategy, LLMConfig

async def generate_optimized_regex():
    """
    Use LLM ONCE to generate optimized regex patterns,
    then use them unlimited times with zero LLM calls.
    """
    cache_file = Path("./patterns/price_patterns.json")
    if cache_file.exists():
        patterns = json.load(cache_file.open())
        print("✅ Using cached regex patterns (FREE)")
    else:
        print("🔄 Generating regex patterns (ONE-TIME LLM COST)...")
        llm_config = LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token="env:OPENAI_API_KEY"
        )
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun("https://example.com/pricing")
            sample_html = result.cleaned_html
        patterns = RegexExtractionStrategy.generate_pattern(
            label="pricing_info",
            html=sample_html,
            query="Extract all pricing information including discounts and special offers",
            llm_config=llm_config
        )
        cache_file.parent.mkdir(exist_ok=True)
        json.dump(patterns, cache_file.open("w"), indent=2)
        print("✅ Patterns generated and cached")
    strategy = RegexExtractionStrategy(custom=patterns)
    return strategy

# Use generated patterns for unlimited extractions
# strategy = await generate_optimized_regex()
EOF

# 09_demo_multistrategy.py
cat > 09_demo_multistrategy.py <<'EOF'
import json
import asyncio
from crawl4ai import AsyncWebCrawler, RegexExtractionStrategy, CrawlerRunConfig, JsonCssExtractionStrategy

async def multi_strategy_pipeline():
    """
    Efficient pipeline using multiple non-LLM strategies:
    1. Regex for simple patterns (fastest)
    2. Schema for structured data
    3. Only use LLM if absolutely necessary
    """
    url = "https://example.com/complex-page"
    async with AsyncWebCrawler() as crawler:
        # Strategy 1: Fast regex for contact info
        regex_strategy = RegexExtractionStrategy(
            pattern=RegexExtractionStrategy.Email | RegexExtractionStrategy.PhoneUS
        )
        regex_config = CrawlerRunConfig(extraction_strategy=regex_strategy)
        regex_result = await crawler.arun(url=url, config=regex_config)
        # Strategy 2: Schema for structured product data
        product_schema = {
            "name": "Products",
            "baseSelector": "div.product",
            "fields": [
                {"name": "name", "selector": "h3", "type": "text"},
                {"name": "price", "selector": ".price", "type": "text"}
            ]
        }
        css_strategy = JsonCssExtractionStrategy(product_schema)
        css_config = CrawlerRunConfig(extraction_strategy=css_strategy)
        css_result = await crawler.arun(url=url, config=css_config)
        results = {
            "contacts": json.loads(regex_result.extracted_content) if regex_result.success else [],
            "products": json.loads(css_result.extracted_content) if css_result.success else []
        }
        print(f"✅ Extracted {len(results['contacts'])} contacts (regex)")
        print(f"✅ Extracted {len(results['products'])} products (schema)")
        return results

asyncio.run(multi_strategy_pipeline())
EOF

# 10_demo_cache_and_optimization.py
cat > 10_demo_cache_and_optimization.py <<'EOF'
import json
from pathlib import Path

# Cache schemas and patterns for maximum efficiency
class ExtractionCache:
    def __init__(self):
        self.schemas = {}
        self.patterns = {}
    def get_schema(self, site_name):
        if site_name not in self.schemas:
            schema_file = Path(f"./cache/{site_name}_schema.json")
            if schema_file.exists():
                self.schemas[site_name] = json.load(schema_file.open())
        return self.schemas.get(site_name)
    def save_schema(self, site_name, schema):
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        schema_file = cache_dir / f"{site_name}_schema.json"
        json.dump(schema, schema_file.open("w"), indent=2)
        self.schemas[site_name] = schema

cache = ExtractionCache()
# Usage: see docs or previous examples

# Optimize selectors for speed
fast_schema = {
    "name": "Optimized Extraction",
    "baseSelector": "#products > .product",
    "fields": [
        {"name": "title", "selector": "> h3", "type": "text"},
        {"name": "price", "selector": ".price:first-child", "type": "text"}
    ]
}

# Avoid slow selectors
slow_schema = {
    "baseSelector": "div div div .product",
    "fields": [
        {"selector": "* h3", "type": "text"}
    ]
}
EOF

echo "Created Crawl4AI structured data extraction usage example files in $EXAMPLES_DIR"
