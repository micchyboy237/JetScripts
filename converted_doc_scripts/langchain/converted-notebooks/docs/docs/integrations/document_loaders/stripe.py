from jet.logger import logger
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import StripeLoader
import os
import shutil


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
# Stripe

>[Stripe](https://stripe.com/en-ca) is an Irish-American financial services and software as a service (SaaS) company. It offers payment-processing software and application programming interfaces for e-commerce websites and mobile applications.

This notebook covers how to load data from the `Stripe REST API` into a format that can be ingested into LangChain, along with example usage for vectorization.
"""
logger.info("# Stripe")


"""
The Stripe API requires an access token, which can be found inside of the Stripe dashboard.

This document loader also requires a `resource` option which defines what data you want to load.

Following resources are available:

`balance_transations` [Documentation](https://stripe.com/docs/api/balance_transactions/list)

`charges` [Documentation](https://stripe.com/docs/api/charges/list)

`customers` [Documentation](https://stripe.com/docs/api/customers/list)

`events` [Documentation](https://stripe.com/docs/api/events/list)

`refunds` [Documentation](https://stripe.com/docs/api/refunds/list)

`disputes` [Documentation](https://stripe.com/docs/api/disputes/list)
"""
logger.info("The Stripe API requires an access token, which can be found inside of the Stripe dashboard.")

stripe_loader = StripeLoader("charges")

index = VectorstoreIndexCreator().from_loaders([stripe_loader])
stripe_doc_retriever = index.vectorstore.as_retriever()

logger.info("\n\n[DONE]", bright=True)