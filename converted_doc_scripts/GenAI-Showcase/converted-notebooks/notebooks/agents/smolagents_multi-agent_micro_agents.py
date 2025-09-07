from bson.objectid import ObjectId
from datetime import datetime
from google.colab import userdata
from jet.logger import CustomLogger
from pymongo import MongoClient
from smolagents import CodeAgent, LiteLLMModel, ManagedAgent, tool
from smolagents.agents import ToolCallingAgent
from typing import Dict, List
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/smolagents_multi-agent_micro_agents.ipynb)

# Multi-Agent Order Management System with MongoDB

This notebook implements a multi-agent system for managing product orders, inventory, and deliveries using:
- [smolagents](https://github.com/huggingface/smolagents/tree/main) for agent management
- MongoDB for data persistence
- DeepSeek Chat as the LLM model

## Setup
First, let's install required dependencies:
"""
logger.info("# Multi-Agent Order Management System with MongoDB")

# !pip install smolagents pymongo litellm

"""
## Import Dependencies
Import all required libraries and setup the LLM model:
"""
logger.info("## Import Dependencies")



MODEL_ID = "deepseek/deepseek-chat"
MONGODB_URI = userdata.get("MONGO_URI")
DEEPSEEK_API_KEY = userdata.get("DEEPSEEK_API_KEY")

"""
## Database Connection Class
Create a MongoDB connection manager:
"""
logger.info("## Database Connection Class")

mongoclient = MongoClient(MONGODB_URI, appname="devrel.showcase.multi-smolagents")
db = mongoclient.warehouse

"""
## Agent Tools Defenitions
Define tools for each agent type:
"""
logger.info("## Agent Tools Defenitions")

@tool
def check_stock(product_id: str) -> Dict:
    """Query product stock level.

    Args:
        product_id: Product identifier

    Returns:
        Dict containing product details and quantity
    """
    return db.products.find_one({"_id": product_id})


@tool
def update_stock(product_id: str, quantity: int) -> bool:
    """Update product stock quantity.

    Args:
        product_id: Product identifier
        quantity: Amount to decrease from stock

    Returns:
        bool: Success status
    """
    result = db.products.update_one(
        {"_id": product_id}, {"$inc": {"quantity": -quantity}}
    )
    return result.modified_count > 0

@tool
def create_order(products: any, address: str) -> str:
    """Create new order for all provided products.

    Args:
        products: List of products with quantities
        address: Delivery address

    Returns:
        str: Order ID message
    """
    order = {
        "products": products,
        "status": "pending",
        "delivery_address": address,
        "created_at": datetime.now(),
    }
    result = db.orders.insert_one(order)
    return f"Successfully ordered : {result.inserted_id!s}"



@tool
def update_delivery_status(order_id: str, status: str) -> bool:
    """Update order delivery status to in_transit once a pending order is provided

    Args:
        order_id: Order identifier
        status: New delivery status is being set to in_transit or delivered

    Returns:
        bool: Success status
    """
    if status not in ["pending", "in_transit", "delivered", "cancelled"]:
        raise ValueError("Invalid delivery status")

    result = db.orders.update_one(
        {"_id": ObjectId(order_id), "status": "pending"}, {"$set": {"status": status}}
    )
    return result.modified_count > 0

"""
## Main Order Management System
Define the main system class that orchestrates all agents:
"""
logger.info("## Main Order Management System")

class OrderManagementSystem:
    """Multi-agent order management system"""

    def __init__(self, model_id: str = MODEL_ID):
        self.model = LiteLLMModel(model_id=model_id, api_key=DEEPSEEK_API_KEY)

        self.inventory_agent = ToolCallingAgent(
            tools=[check_stock, update_stock], model=self.model, max_iterations=10
        )

        self.order_agent = ToolCallingAgent(
            tools=[create_order], model=self.model, max_iterations=10
        )

        self.delivery_agent = ToolCallingAgent(
            tools=[update_delivery_status], model=self.model, max_iterations=10
        )

        self.managed_agents = [
            ManagedAgent(
                self.inventory_agent, "inventory", "Manages product inventory"
            ),
            ManagedAgent(self.order_agent, "orders", "Handles order creation"),
            ManagedAgent(self.delivery_agent, "delivery", "Manages delivery status"),
        ]

        self.manager = CodeAgent(
            tools=[],
            system_prompt="""For each order:
            1. Create the order document
            2. Update the inventory
            3. Set deliviery status to in_transit

            Use relevant agents:  {{managed_agents_descriptions}}  and you can use {{authorized_imports}}
            """,
            model=self.model,
            managed_agents=self.managed_agents,
            additional_authorized_imports=["time", "json"],
        )

    def process_order(self, orders: List[Dict]) -> str:
        """Process a set of orders.

        Args:
            orders: List of orders each has address and products

        Returns:
            str: Processing result
        """
        return self.manager.run(
            f"Process the following  {orders} as well as substract the ordered items from inventory."
            f"to be delivered to relevant addresses"
        )

"""
## Adding Sample Data
To test the system, you might want to add some sample products to MongoDB:
"""
logger.info("## Adding Sample Data")

def add_sample_products():
    db.products.delete_many({})
    sample_products = [
        {"_id": "prod1", "name": "Laptop", "price": 999.99, "quantity": 10},
        {"_id": "prod2", "name": "Smartphone", "price": 599.99, "quantity": 15},
        {"_id": "prod3", "name": "Headphones", "price": 99.99, "quantity": 30},
    ]

    db.products.insert_many(sample_products)
    logger.debug("Sample products added successfully!")


add_sample_products()

"""
## Testing the System
Let's test our system with a sample order:
"""
logger.info("## Testing the System")

system = OrderManagementSystem()

test_orders = [
    {
        "products": [
            {"product_id": "prod1", "quantity": 2},
            {"product_id": "prod2", "quantity": 1},
        ],
        "address": "123 Main St",
    },
    {"products": [{"product_id": "prod3", "quantity": 3}], "address": "456 Elm St"},
]

result = system.process_order(orders=test_orders)

logger.debug("Orders processing result:", result)

"""
## Conclusions
In this notebook, we have successfully implemented a multi-agent order management system using smolagents and MongoDB. We defined various tools for managing inventory, creating orders, and updating delivery statuses. We also created a main system class to orchestrate these agents and tested the system with sample data and orders.

This approach demonstrates the power of combining agent-based systems with robust data persistence solutions like MongoDB to create scalable and efficient order management systems.
"""
logger.info("## Conclusions")

logger.info("\n\n[DONE]", bright=True)