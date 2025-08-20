from jet.logger import CustomLogger
from llama_index.readers.document360 import Document360Reader
from llama_index.readers.document360.entities import (
Article,
ArticleSlim,
)
from llama_index.readers.document360.entities import (
Article,
ArticleSlim,
Category,
ProjectVersion,
)
from llama_index.readers.document360.entities import Article
from llama_index.readers.document360.entities import Article, Category
from your_module import ProcessedArticle, LinkExtractor
import logging
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Document360

## Simple Example
"""
logger.info("# Document360")


api_key = "document360_api_key"

reader = Document360Reader(api_key=api_key)

documents = reader.load_data()

for d in documents:
    logger.debug(d.text)

"""
## Customize Document360Reader Example

### Filter entities to process
"""
logger.info("## Customize Document360Reader Example")





def should_process_project_version(project_version: ProjectVersion):
    project_versions_of_interest = ["document360_project_version_id"]

    return project_version.get_id() in project_versions_of_interest:

def should_process_category(
    category: Category, parent_categories: list[Category]
):
    categories_of_interest = ["document360_category_id"]

    return category.get_id() in categories_of_interest

def should_process_article(article: ArticleSlim):
    return article.get_title() !== "Do not process this article"


reader = Document360Reader(
    api_key=api_key,
    should_process_project_version=should_process_project_version,
    should_process_category=should_process_category,
    should_process_article=should_process_article,
)

reader.load_data()

"""
### Customizing Error Handling
"""
logger.info("### Customizing Error Handling")





def handle_rate_limit_error():
    logging.error("Rate limit exceeded. Retrying...")


def handle_request_http_error(e: Exception):
    logging.error(f"HTTP Request failed. {e}")


def handle_article_processing_error(e: Exception, article: Union[Article, ArticleSlim]):
    logging.error(f"Failed to process {article}: {e}")


def handle_load_data_error(e: Exception):
    logging.error(f"Load data error: {e}")


reader = Document360Reader(
    api_key=api_key,
    handle_rate_limit_error=handle_rate_limit_error,
    handle_request_http_error=handle_request_http_error,
    handle_article_processing_error=handle_article_processing_error,
    handle_load_data_error=handle_load_data_error,
)

reader.load_data()

"""
### Hook into the Document360Reader Lifecycle
"""
logger.info("### Hook into the Document360Reader Lifecycle")





def handle_batch_finished():
    logging.info("Batch finished processing")


def handle_category_processing_started(category: Category):
    logging.info(f"Started processing category: {category}")


def handle_article_processing_started(article: Article):
    logging.info(f"Processing article: {article}")


reader = Document360Reader(
    api_key=api_key,
    handle_batch_finished=handle_batch_finished,
    handle_category_processing_started=handle_category_processing_started,
    handle_article_processing_started=handle_article_processing_started,
)

reader.load_data()

"""
### Create a custom llama_index Document from Document360 Article
"""
logger.info("### Create a custom llama_index Document from Document360 Article")





def article_to_custom_document(article: Article):
    processed_article = ProcessedArticle(article=article)

    processed_article.extract_links(LinkExtractor())
    links = processed_article.get_links()

    return Document(
        doc_id=article.get_id(),
        text=strip_html(article.get_html_content()),
        extra_info={
            "title": article.get_title(),
            "category_id": article.get_category_id(),
            "project_version_id": article.get_project_version_id(),
            "created_by": article.get_created_by(),
            "created_at": article.get_created_at(),
            "modified_at": article.get_modified_at(),
            "url": article.get_url(),
            "links": links,
        },
    )


reader = Document360Reader(
    api_key=api_key,
    article_to_custom_document=article_to_custom_document,
)

reader.load_data()

logger.info("\n\n[DONE]", bright=True)