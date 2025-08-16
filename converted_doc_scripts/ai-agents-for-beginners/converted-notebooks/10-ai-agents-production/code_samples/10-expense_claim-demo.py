import asyncio
from jet.transformers.formatters import format_json
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, TextContentItem, ImageContentItem, ImageUrl, ImageDetailLevel
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from jet.logger import CustomLogger
from openai import AsyncOllama
from pydantic import BaseModel, Field
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.agents.strategies import SequentialSelectionStrategy, DefaultTerminationStrategy
from semantic_kernel.connectors.ai.open_ai import OllamaChatCompletion, OllamaChatPromptExecutionSettings
from semantic_kernel.contents import ImageContent, TextContent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.kernel import Kernel
from typing import List
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Expense Claim Analysis

This notebook demonstrates how to create agents that use plugins to process travel expenses from local receipt images, generate an expense claim email, and visualize expense data using a pie chart. Agents dynamically choose functions based on the task context.

Steps:
1. OCR Agent processes the local receipt image and extracts travel expense data.
2. Email Agent generates an expense claim email.

### Example of a travel expense scenario:
Imagine you're an employee traveling for a business meeting in another city. Your company has a policy to reimburse all reasonable travel-related expenses. Here’s a breakdown of potential travel expenses:
- Transportation:
Airfare for a round trip from your home city to the destination city.
Taxi or ride-hailing services to and from the airport.
Local transportation in the destination city (like public transit, rental cars, or taxis).

- Accommodation:
Hotel stay for three nights at a mid-range business hotel close to the meeting venue.

- Meals:
Daily meal allowance for breakfast, lunch, and dinner, based on the company's per diem policy.

- Miscellaneous Expenses:
Parking fees at the airport.
Internet access charges at the hotel.
Tips or small service charges.

- Documentation:
You submit all receipts (flights, taxis, hotel, meals, etc.) and a completed expense report for reimbursement.

## Import required libraries

Import the necessary libraries and modules for the notebook.
"""
logger.info("# Expense Claim Analysis")





load_dotenv()

def _create_kernel_with_chat_completion(service_id: str) -> Kernel:
    kernel = Kernel()

    client = AsyncOllama(
    api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.inference.ai.azure.com/")
    kernel.add_service(
        OllamaChatCompletion(
            ai_model_id="llama3.1",
            async_client=client,
            service_id="open_ai"
        )
    )

    kernel.add_service(
        OllamaChatCompletion(
            ai_model_id="gpt-4o",
            async_client=client,
            service_id="gpt-4o"
        )
    )

    return kernel

"""
## Define Expense Models

 Create a Pydantic model for individual expenses and an ExpenseFormatter class to convert a user query into structured expense data.

 Each expense will be represented in the format:
 `{'date': '07-Mar-2025', 'description': 'flight to destination', 'amount': 675.99, 'category': 'Transportation'}`
"""
logger.info("## Define Expense Models")

class Expense(BaseModel):
    date: str = Field(..., description="Date of expense in dd-MMM-yyyy format")
    description: str = Field(..., description="Expense description")
    amount: float = Field(..., description="Expense amount")
    category: str = Field(..., description="Expense category (e.g., Transportation, Meals, Accommodation, Miscellaneous)")

class ExpenseFormatter(BaseModel):
    raw_query: str = Field(..., description="Raw query input containing expense details")

    def parse_expenses(self) -> List[Expense]:
        """
        Parses the raw query into a list of Expense objects.
        Expected format: "date|description|amount|category" separated by semicolons.
        """
        expense_list = []
        for expense_str in self.raw_query.split(";"):
            if expense_str.strip():
                parts = expense_str.strip().split("|")
                if len(parts) == 4:
                    date, description, amount, category = parts
                    try:
                        expense = Expense(
                            date=date.strip(),
                            description=description.strip(),
                            amount=float(amount.strip()),
                            category=category.strip()
                        )
                        expense_list.append(expense)
                    except ValueError as e:
                        logger.debug(f"[LOG] Parse Error: Invalid data in '{expense_str}': {e}")
        return expense_list

"""
## Defining Agents - Generating the Email

Create an agent class to generate an email for submitting an expense claim.
- This agent uses the `kernel_function` decorator to define a function that generates an email for submitting an expense claim.
- It calculates the total amount of the expenses and formats the details into an email body.
"""
logger.info("## Defining Agents - Generating the Email")

class ExpenseEmailAgent:

    @kernel_function(description="Generate an email to submit an expense claim to the Finance Team")
    async def generate_expense_email(expenses):
        total_amount = sum(expense['amount'] for expense in expenses)
        email_body = "Dear Finance Team,\n\n"
        email_body += "Please find below the details of my expense claim:\n\n"
        for expense in expenses:
            email_body += f"- {expense['description']}: ${expense['amount']}\n"
        email_body += f"\nTotal Amount: ${total_amount}\n\n"
        email_body += "Receipts for all expenses are attached for your reference.\n\n"
        email_body += "Thank you,\n[Your Name]"
        return email_body

"""
# Agent for Extracting Travel Expenses from Receipt Images

Create an agent class to extract travel expenses from receipt images.
- This agent uses the `kernel_function` decorator to define a function that extracts travel expenses from receipt images.
- Convert the receipt image to text using OCR (Optical Character Recognition) and extract relevant information such as date, description, amount, and category.
"""
logger.info("# Agent for Extracting Travel Expenses from Receipt Images")

class OCRAgentPlugin:
    def __init__(self):
        self.client = ChatCompletionsClient(
            endpoint="https://models.inference.ai.azure.com/",
            credential=AzureKeyCredential(os.environ.get("GITHUB_TOKEN")),
        )
        self.model_name = "gpt-4o"

    @kernel_function(description="Extract structured travel expense data from receipt.jpg using gpt-4o-model")
    def extract_text(self, image_path: str = "receipt.jpg") -> str:
        try:
            image_url_str = str(ImageUrl.load(image_file=image_path, image_format="jpg", detail=ImageDetailLevel.HIGH))

            prompt = (
                "You are an expert OCR assistant specialized in extracting structured data from receipt images. "
                "Analyze the provided receipt image and extract travel-related expense details in the format: "
                "'date|description|amount|category' separated by semicolons. "
                "Follow these rules: "
                "- Date: Convert dates (e.g., '4/4/22') to 'dd-MMM-yyyy' (e.g., '04-Apr-2022'). "
                "- Description: Extract item names (e.g., 'Carlson's Drylawn', 'Peigs transaction Probiotics'). "
                "- Amount: Use numeric values (e.g., '4.50' from '$4.50' or '4.50 dollars'). "
                "- Category: Infer from context (e.g., 'Meals' for food, 'Transportation' for travel, 'Accommodation' for lodging, 'Miscellaneous' otherwise). "
                "Ignore totals, subtotals, or service charges unless they are itemized expenses. "
                "If no expenses are found, return 'No expenses detected'. "
                "Return only the structured data, no additional text."
            )
            response = self.client.complete(
                messages=[
                    SystemMessage(content=prompt),
                    UserMessage(content=[
                        TextContentItem(text="Extract travel expenses from this receipt image."),
                        ImageContentItem(image_url=ImageUrl(url=image_url_str))
                    ])
                ],
                model=self.model_name,
                temperature=0.1,
                max_tokens=2048
            )
            extracted_text = response.choices[0].message.content
            return extracted_text
        except Exception as e:
            error_msg = f"[LOG] OCR Plugin: Error processing image: {str(e)}"
            logger.debug(error_msg)
            return error_msg

"""
## Processing Expenses

Define an asynchronous function to process the expenses by creating and registering the necessary agents and then invoking them.
- This function processes the expenses by loading environment variables, creating the necessary agents, and registering them as plugins.
- It creates a group chat with the two agents and sends a prompt message to generate the email and pie chart based on the expenses data.
- It handles any errors that occur during the chat invocation and ensures proper cleanup of the agents.
"""
logger.info("## Processing Expenses")

async def process_expenses():
    load_dotenv()
    settings_slm = OllamaChatPromptExecutionSettings(service_id="gpt-4o")
    settings_llm = OllamaChatPromptExecutionSettings(service_id="open_ai")  # Fixed typo in service_id

    ocr_agent = ChatCompletionAgent(
        kernel=_create_kernel_with_chat_completion("ocrAgent"),
        name="ocr_agent",
        instructions="Extract travel expense data from the receipt image in the prompt using the 'extract_text' function from the 'ocrAgent' plugin. Return the data in the format 'date|description|amount|category' separated by semicolons.",
        arguments=KernelArguments(settings=settings_slm)
    )


    email_agent = ChatCompletionAgent(
            kernel=_create_kernel_with_chat_completion("expenseEmailAgent"),
            name="email_agent",
            instructions="Take the travel expense data from the previous agent and generate a professional expense claim email using the 'generate_expense_email' function from the 'expenseEmailAgent' plugin, then pass the data forward.",
            arguments=KernelArguments(
                settings=settings_llm)
        )


    kernel = Kernel()

    image_path = "./receipt.jpg"

    image_url_str = f"file://{image_path}"

    user_message = ChatMessageContent(
        role=AuthorRole.USER,
        items=[
            TextContent(text="""
            Please extract the raw text from this receipt image, focusing on travel expenses like dates, descriptions, amounts, and categories (e.g., Transportation, Accommodation, Meals, Miscellaneous).
            Then generate a professional expense claim email.
                        """),
            ImageContent.from_image_file(path=image_path)
        ]
    )

    kernel.add_plugin(OCRAgentPlugin(), plugin_name="ocrAgent")
    kernel.add_plugin(ExpenseEmailAgent(), plugin_name="expenseEmailAgent")

    chat = AgentGroupChat(
        agents=[ocr_agent, email_agent],
        selection_strategy=SequentialSelectionStrategy(initial_agent=ocr_agent),
        termination_strategy=DefaultTerminationStrategy(maximum_iterations=1)
    )

    async def run_async_code_d471069f():
        await chat.add_chat_message(user_message)
        return 
     = asyncio.run(run_async_code_d471069f())
    logger.success(format_json())
    logger.debug(f"# User message added to chat with receipt image")

    async for content in chat.invoke():
        logger.debug(f"# Agent - {content.name or '*'}: '{content.content}'")

"""
## Main function

Define the main function to clear the console and run the `process_expenses` function asynchronously.
"""
logger.info("## Main function")

async def main():
    os.system('cls' if os.name=='nt' else 'clear')

    async def run_async_code_e313b407():
        await process_expenses()
        return 
     = asyncio.run(run_async_code_e313b407())
    logger.success(format_json())

async def run_async_code_ba09313d():
    await main()
    return 
 = asyncio.run(run_async_code_ba09313d())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)