"""
Sample Python Module for Unstructured Testing
=============================================
This file demonstrates how Unstructured handles Python source code.
It includes docstrings, classes, functions, and inline comments.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InvoiceItem:
    """Represents a single line item on an invoice."""

    description: str
    quantity: int
    unit_price: float

    @property
    def total(self) -> float:
        """Calculate line item total."""
        return self.quantity * self.unit_price


def calculate_grand_total(items: List[InvoiceItem]) -> float:
    """
    Sum all line item totals.

    Args:
        items: List of InvoiceItem objects

    Returns:
        Grand total as float
    """
    if not items:
        logger.warning("Empty item list provided")
        return 0.0
    return sum(item.total for item in items)


class DocumentProcessor:
    """Process documents through multiple stages."""

    SUPPORTED_FORMATS = ["pdf", "docx", "html"]

    def __init__(self, max_pages: Optional[int] = None):
        self.max_pages = max_pages
        self._processed_count = 0

    def process(self, file_path: str) -> dict:
        """Main processing entry point."""
        logger.info(f"Processing {file_path}")
        # TODO: Add actual processing logic
        self._processed_count += 1
        return {"status": "success", "pages": self._processed_count}


if __name__ == "__main__":
    sample_items = [
        InvoiceItem("Widget A", 5, 12.99),
        InvoiceItem("Gadget B", 2, 45.50),
    ]
    total = calculate_grand_total(sample_items)
    print(f"Grand Total: ${total:.2f}")
