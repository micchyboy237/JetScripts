from jet.logger import logger
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import ModernTreasuryLoader
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
# Modern Treasury

>[Modern Treasury](https://www.moderntreasury.com/) simplifies complex payment operations. It is a unified platform to power products and processes that move money.
>- Connect to banks and payment systems
>- Track transactions and balances in real-time
>- Automate payment operations for scale

This notebook covers how to load data from the `Modern Treasury REST API` into a format that can be ingested into LangChain, along with example usage for vectorization.
"""
logger.info("# Modern Treasury")


"""
The Modern Treasury API requires an organization ID and API key, which can be found in the Modern Treasury dashboard within developer settings.

This document loader also requires a `resource` option which defines what data you want to load.

Following resources are available:

`payment_orders` [Documentation](https://docs.moderntreasury.com/reference/payment-order-object)

`expected_payments` [Documentation](https://docs.moderntreasury.com/reference/expected-payment-object)

`returns` [Documentation](https://docs.moderntreasury.com/reference/return-object)

`incoming_payment_details` [Documentation](https://docs.moderntreasury.com/reference/incoming-payment-detail-object)

`counterparties` [Documentation](https://docs.moderntreasury.com/reference/counterparty-object)

`internal_accounts` [Documentation](https://docs.moderntreasury.com/reference/internal-account-object)

`external_accounts` [Documentation](https://docs.moderntreasury.com/reference/external-account-object)

`transactions` [Documentation](https://docs.moderntreasury.com/reference/transaction-object)

`ledgers` [Documentation](https://docs.moderntreasury.com/reference/ledger-object)

`ledger_accounts` [Documentation](https://docs.moderntreasury.com/reference/ledger-account-object)

`ledger_transactions` [Documentation](https://docs.moderntreasury.com/reference/ledger-transaction-object)

`events` [Documentation](https://docs.moderntreasury.com/reference/events)

`invoices` [Documentation](https://docs.moderntreasury.com/reference/invoices)
"""
logger.info("The Modern Treasury API requires an organization ID and API key, which can be found in the Modern Treasury dashboard within developer settings.")

modern_treasury_loader = ModernTreasuryLoader("payment_orders")

index = VectorstoreIndexCreator().from_loaders([modern_treasury_loader])
modern_treasury_doc_retriever = index.vectorstore.as_retriever()

logger.info("\n\n[DONE]", bright=True)