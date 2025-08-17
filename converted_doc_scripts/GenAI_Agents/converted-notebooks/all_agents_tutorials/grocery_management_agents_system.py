from crewai import Agent, Task, Crew
from crewai_tools import WebsiteSearchTool
from jet.logger import CustomLogger
from markdown import markdown
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# 🛒 Grocery Management Agents System

This tutorial will guide you through using CrewAI agents to automate grocery management. We'll cover how to extract grocery data from receipts, estimate expiration dates, track grocery inventory, and recommend recipes using leftover items.
🎥 Youtube video: [Hackathon Grocery Management Agents System - Disha An](https://youtu.be/F1vN8vclpGM)


## 📋 Table of Contents
1. [Project Workflow](#work-flow) 
2. [Environment Setup](#environment-setup)
3. [Reading the Receipt](#reading-the-receipt)
4. [Creating the Agents](#creating-the-agents)
   - Receipt Interpreter Agent
   - Expiration Date Estimation Agent
   - Grocery Tracker Agent
   - Recipe Recommendation Agent
5. [Defining the Tasks](#defining-the-tasks)
   - Task for Reading the Receipt
   - Task for Expiration Date Estimation
   - Task for Grocery Tracking
   - Task for Recipe Recommendation
6. [Running the Crew](#running-the-crew)

## 🔄 1. Project Workflow <a id="work-flow"></a>
![Grocery Management Agents System Workflow](../images/grocery_management_agents_system.png)

## 🌐 2. Environment Setup <a id="environment-setup"></a>

### Step 1: Install Required Packages
Make sure you have the necessary packages installed:
"""
logger.info("# 🛒 Grocery Management Agents System")

# !pip install Markdown==3.7
# !pip install crewai==0.80.0
# !pip install crewai_tools

"""
### Step 2: Set Up Your API Key
You will need an MLX API key to proceed. Please store it securely and load it into your environment.

Additionally, if you wish to test the functionality that reads real receipts and converts them into markdown files, you'll need a LLAMA OCR API key. This is optional but recommended for testing with actual receipt images. You can obtain a LLAMA OCR API key from [here](https://api.together.ai/).

Note: Sample receipts have already been processed and saved in the file located at:
`data/grocery_management_agents_system/extracted/grocery_receipt.md`.
"""
logger.info("### Step 2: Set Up Your API Key")


# os.environ["OPENAI_API_KEY"] = "[YOUR OPENAI API KEY]"

os.environ["LLAMA_OCR_API_KEY"] = "[YOUR LLAMA OCR API KEY]"

"""
### Step 3: Extract Receipt Information from a Receipt Image (Optional)

By default, the test extracted information has already been saved in:  
`GenAI_Agents/data/grocery_management_agents_system/extracted/grocery_receipt.md`.

However, if you'd like to test using different receipt images, you can do so by following these steps:

1. **Add Your Receipt Image**  
   Place your image in the following folder:  
   `GenAI_Agents/data/grocery_management_agents_system/input`

2. **Update the Script**  
   Open the `extract_items.js` file and change the `filePath` variable to the name of your new image.

3. **Run the Script**  
   In your terminal, navigate to the input directory and run the script:

   ```bash
   cd GenAI_Agents/data/grocery_management_agents_system/input
   node extract_items.js
The newly generated markdown file will be saved in:
`GenAI_Agents/data/grocery_management_agents_system/extracted/`

**How to Use Node.js**

To get started with Node.js, you'll first need to install **NVM (Node Version Manager)**. This allows you to easily manage different versions of Node.js on your system.

For macOS users, you can find a detailed guide on installing NVM [here](https://medium.com/@andrewjaykeller/how-to-install-node-js-and-npm-with-macoss-new-terminal-zsh-e39b4a62d3d4).

## 🧾 3. Reading the Receipt <a id="reading-the-receipt"></a>
We'll start by reading a markdown file containing the grocery receipt.
"""
logger.info("### Step 3: Extract Receipt Information from a Receipt Image (Optional)")


with open('../data/grocery_management_agents_system/extracted/grocery_receipt.md', 'r') as f:
    receipt_markdown = markdown(f.read())

today = "2024-11-16"
logger.debug("Receipt loaded successfully!")

"""
## 🤖 4. Creating the Agents <a id="creating-the-agents"></a>
### Step 4.1: Receipt Interpreter Agent
This agent extracts item details from the receipt, such as names, quantities, and units.
"""
logger.info("## 🤖 4. Creating the Agents <a id="creating-the-agents"></a>")

receipt_interpreter_agent = Agent(
    role="Receipt Markdown Interpreter",
    goal=(
        "Accurately extract items, their counts, and weights with units from a given receipt in markdown format. "
        "Provide structured data to support the grocery management system."
    ),
    backstory=(
        "As a key member of the grocery management crew for the household, your mission is to meticulously extract "
        "details such as item names, quantities, and weights from receipt markdown files. Your role is vital for the "
        "grocery tracker agent, which monitors the household's inventory levels."
    ),
    personality=(
        "Diligent, detail-oriented, and efficient. The Receipt Markdown Interpreter is committed to providing accurate "
        "and structured information to support effective grocery management. It is particularly focused on clarity and precision."
    ),
    allow_delegation=False,
    verbose=True
)

"""
### Step 4.2: Expiration Date Estimation Agent
This agent estimates the expiration dates of items using an online source.
"""
logger.info("### Step 4.2: Expiration Date Estimation Agent")

expiration_date_search_web_tool = WebsiteSearchTool(website='https://www.stilltasty.com/')

expiration_date_search_agent = Agent(
    role="Expiration Date Estimation Specialist",
    goal=(
        "Accurately estimate the expiration dates of items extracted by the Receipt Markdown Interpreter Agent. "
        "Utilize online sources to determine typical shelf life when refrigerated and add the estimated number of days to the purchase date."
    ),
    backstory=(
        "As the Expiration Date Estimation Specialist, your role is to ensure the household's groceries are consumed before expiration. "
        "You use your access to online resources to search for the best estimates on how long each item typically lasts when stored properly."
    ),
    personality=(
        "Meticulous, resourceful, and reliable. This agent ensures the household maintains a well-stocked but efficiently used inventory, minimizing waste."
    ),
    allow_delegation=False,
    verbose=True,
    tools=[expiration_date_search_web_tool]
)

"""
### Step 4.3: Grocery Tracker Agent
Tracks the remaining inventory based on user input.
"""
logger.info("### Step 4.3: Grocery Tracker Agent")

grocery_tracker_agent = Agent(
    role="Grocery Inventory Tracker",
    goal=(
        "Accurately track the remaining groceries based on user consumption input. "
        "Subtract consumed items from the grocery list obtained from the Expiration Date Estimation Specialist and update the inventory. "
        "Provide the user with an updated list of what's left, along with corresponding expiration dates."
    ),
    backstory=(
        "As the household's Grocery Inventory Tracker, your responsibility is to ensure that groceries are accurately tracked based on user input. "
        "You need to understand the user's input on what they've consumed, update the inventory list, and remind them of what's left and the expiration dates. "
        "Your role is crucial in helping the household avoid waste and ensure timely consumption of perishable items."
    ),
    personality=(
        "Helpful, detail-oriented, and responsive. This agent is focused on ensuring the household has an up-to-date inventory, minimizing waste, and helping users stay organized."
    ),
    allow_delegation=False,
    verbose=True
)

"""
### Step 4.4: Recipe Recommendation Agent
Suggests recipes based on the remaining groceries.
"""
logger.info("### Step 4.4: Recipe Recommendation Agent")

recipe_web_tool = WebsiteSearchTool(website='https://www.americastestkitchen.com/recipes')

rest_grocery_recipe_agent = Agent(
    role="Grocery Recipe Recommendation Specialist",
    goal=(
        "Provide recipe recommendations using the remaining groceries in the inventory. "
        "Avoid using items with a count of 0 and prioritize recipes that maximize the use of available ingredients. "
        "If ingredients are insufficient, suggest restocking recommendations."
    ),
    backstory=(
        "As a Grocery Recipe Recommendation Specialist, your mission is to help the household make the most out of their remaining groceries. "
        "Your role is to search the web for easy, delicious recipes that utilize available ingredients while minimizing waste. "
        "Ensure that the recipes are simple to follow and use as many of the remaining ingredients as possible."
    ),
    personality=(
        "Creative, resourceful, and efficient. This agent is dedicated to helping the household create enjoyable meals with what they have on hand."
    ),
    allow_delegation=False,
    verbose=True,
    tools=[recipe_web_tool],
    human_input=True
)

"""
## 📝 5. Defining the Tasks <a id="defining-the-tasks"></a>
### Step 5.1: Task for Reading the Receipt
This task extracts item details from the receipt.
"""
logger.info("## 📝 5. Defining the Tasks <a id="defining-the-tasks"></a>")

read_receipt_task = Task(
    agent=receipt_interpreter_agent,
    description=(
        f"Analyze the receipt markdown file provided: {receipt_markdown}. "
        "Extract information on items purchased, their counts, weights, and units. "
        f"Additionally, extract today's date information which is provided here: {today}. "
        "Ensure all item names are converted into clear, human-readable text."
    ),
    expected_output="""
    {
        "items": [
            {
                "item_name": "string - Human-readable name of the item",
                "count": "integer - Number of units purchased",
                "unit": "string - Unit of measurement (e.g., kg, lbs, pcs)"
            }
        ],
        "date_of_purchase": "string - Date in YYYY-MM-DD format"
    }
    """
)

"""
### Step 5.2: Task for Expiration Date Estimation
This task estimates expiration dates based on item data.
"""
logger.info("### Step 5.2: Task for Expiration Date Estimation")

expiration_date_search_task = Task(
    agent=expiration_date_search_agent,
    description=(
        "Using the list of items extracted by the Receipt Markdown Interpreter Agent, search online to find the typical shelf life of each item when refrigerated. "
        "Add this information to the date of purchase to estimate the expiration date for each item."
        "Ensure that the output includes the item name, count, unit, and estimated expiration date."
    ),
    expected_output="""
    {
        "items": [
            {
                "item_name": "string - Human-readable name of the item",
                "count": "integer - Number of units purchased",
                "unit": "string - Unit of measurement (e.g., kg, lbs, pcs)",
                "expiration_date": "string - Estimated expiration date in YYYY-MM-DD format"
            }
        ]
    }
    """,
    context=[read_receipt_task]
)

"""
### Step 5.3: Task for Grocery Tracking
This task updates the grocery list based on user input.
"""
logger.info("### Step 5.3: Task for Grocery Tracking")

grocery_tracking_task = Task(
    agent=grocery_tracker_agent,
    description=(
        "Using the grocery list with expiration dates provided by the Expiration Date Estimation Specialist, "
        "update the inventory based on user input about items they have consumed. "
        "Subtract the consumed quantities from the inventory list and provide a summary of what items are left, including their expiration dates. "
        "Ensure that the updated list is returned in JSON format."
    ),
    expected_output="""
    {
        "items": [
            {
                "item_name": "string - Human-readable name of the item",
                "count": "integer - Updated number of units remaining",
                "unit": "string - Unit of measurement (e.g., kg, lbs, pcs)",
                "expiration_date": "string - Estimated expiration date in YYYY-MM-DD format"
            }
        ]
    }
    """,
    context=[expiration_date_search_task],
    human_input=True,
    output_file = "../data/grocery_management_agents_system/output/grocery_tracker.json"
)

"""
### Step 5.4: Task for Recipe Recommendation
This task suggests recipes using available ingredients.
"""
logger.info("### Step 5.4: Task for Recipe Recommendation")

recipe_recommendation_task = Task(
    agent=rest_grocery_recipe_agent,
    description=(
        "Using the updated grocery list provided by the Grocery Inventory Tracker, "
        "search online for recipes that utilize the available ingredients. "
        "Only include items with a count greater than zero. If no suitable recipe can be found, provide restocking recommendations. "
        "Ensure that the output includes recipe names, ingredients, instructions, and the source website."
    ),
    expected_output="""
    {
        "recipes": [
            {
                "recipe_name": "string - Name of the recipe",
                "ingredients": [
                    {
                        "item_name": "string - Ingredient name",
                        "quantity": "string - Quantity required",
                        "unit": "string - Measurement unit (e.g., kg, pcs, tbsp)"
                    }
                ],
                "steps": [
                    "string - Step-by-step instructions for the recipe"
                ],
                "source": "string - Website URL for the recipe"
            }
        ],
        "restock_recommendations": [
            {
                "item_name": "string - Name of the item to restock",
                "quantity_needed": "integer - Suggested quantity to purchase",
                "unit": "string - Measurement unit (e.g., kg, pcs)"
            }
        ]
    }
    """,
    context=[grocery_tracking_task],
    output_file = "../data/grocery_management_agents_system/output/recipe_recommendation.json"
)

"""
## 🚀 6. Running the Crew <a id="running-the-crew"></a>
Now, let's put everything together and run the crew.
"""
logger.info("## 🚀 6. Running the Crew <a id="running-the-crew"></a>")

crew = Crew(agents=[receipt_interpreter_agent,
                    expiration_date_search_agent,
                    grocery_tracker_agent,
                    rest_grocery_recipe_agent],
            tasks=[read_receipt_task,
                   expiration_date_search_task,
                   grocery_tracking_task,
                   recipe_recommendation_task],
            verbose=True)

result = crew.kickoff()

"""
The output for the **Grocery Tracker** and **Recipe Recommendations** is saved in the following directory:  
`data/grocery_management_agents_system/output`


"""
logger.info("The output for the **Grocery Tracker** and **Recipe Recommendations** is saved in the following directory:")

logger.info("\n\n[DONE]", bright=True)