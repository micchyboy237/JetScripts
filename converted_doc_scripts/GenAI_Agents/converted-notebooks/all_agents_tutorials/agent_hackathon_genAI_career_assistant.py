from IPython.display import display, Image, Markdown
from datetime import datetime
from dotenv import load_dotenv
from jet.logger import CustomLogger
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchResults #searching tools
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import MermaidDrawMethod # to visualize the graph of langgraph node and edges
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START #Importing StateGraph, END, and START from langgraph.graph to define and manage state transitions within a conversational or generative AI workflow.
from typing import Dict, TypedDict
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# GenAI Career Assistant Agent – Your Ultimate Guide to a Career in Generative AI!🚀
<a href="https://colab.research.google.com/drive/19lYWHVgRSSyY84HW3WkwJ9OFAA25T8KW?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
## Overview
Meet the GenAI Career Assistant—an AI-powered mentor designed to simplify and support your journey in Generative AI learning, Resume preparation, Interview assistant and job hunting.
#### Tech Stack
I have used all free Open source.<br>
Langchain,Langgraph, Gemini LLM, DuckDuckGoSearchResult

## Motivation
As GenAI rapidly evolves, more people are eager to learn it for career advancement or transition. However, navigating the vast resources on the internet and platforms like YouTube can be overwhelming, with long videos and scattered, outdated materials making it hard to know where to begin in this busy life. Even using ChatGPT for coding help often yields deprecated code, as GenAI packages and methods—such as LangChain, LlamaIndex, and Hugging Face—are updated frequently.

### Key Features

1. **Learning & Content Creation:**
   - Offers tailored learning pathways in GenAI, covering key topics and skills.
   - Assists users in creating tutorials, blogs, and posts based on their interests or queries.
2. **Q&A Support:**
   - Provides on-demand Q&A sessions for users needing guidance on concepts or coding issues.
3. **Resume Building & Review:**
   - One-on-one resume consultations and guidance.
   - Crafts personalized, market-relevant resumes optimized for current job trends.
4. **Interview Preparation:**
   - Hosts Q&A sessions on common and technical interview questions.
   - Simulates real interview scenarios and conducts mock interviews.
5. **Job Search Assistance:**
   - Guides users through the job search process, offering tailored insights and support. 
With the GenAI Career Assistant, your journey to a career in Generative AI becomes organized, personalized, and efficient!

<img src="https://i.imghippo.com/files/xrJV7042k.png" alt="agent" border="0" style="height:20%;width:90%">

## Key Components
1. **State Management**: Using TypedDict to define and manage the state of each customer interaction.
2. **Query Categorization**: Classifying users queries into Learning, Resume Preparation, Interview or Job Search.
3. **Sub Categorization**: Learning(Tutorial, Q&A), Interview(Interview prep,Mock interview).
4. **Response Generation**: Creating appropriate responses based on the query category. Create .md files for Tutorial Blogs, Resume, Mock interview etc.
6. **Workflow Graph**: Utilizing LangGraph to create a flexible and extensible workflow.

## Method Details
1. **Initialization**: Set up the environment and import necessary libraries.
2. **State Definition**: Create a structure to hold query information, category, sub-category, and response.
3. **Node Functions**: Implement separate functions for categorization, and response generation.
4. **Graph Construction**: Use StateGraph to define the workflow, adding nodes and edges to represent the support process.
5. **Conditional Routing**: Implement logic to route queries based on their category and sub- category.
6. **Workflow Compilation**: Compile the graph into an executable application.
7. **Execution**: Process users queries through the workflow and retrieve results.

## Conclusion
The GenAI Career Assistant is more than just a tool; it’s a comprehensive, personalized mentor designed to help you thrive in the rapidly evolving field of Generative AI. From mastering key concepts and building a strong resume to preparing for interviews and navigating the job market, this assistant equips you with everything you need to achieve your career goals. With GenAI Career Assistant by your side, your path to a successful Generative AI career becomes clearer, more manageable, and achievable. Embrace the future of AI with confidence and step into your dream role!
## Future Enhancement
- **Knowledge Base**: Incorporate a resource-rich library with curated links to courses, tutorials, and articles for comprehensive learning support.
- **Multi-Domain Customization**: Expand beyond Generative AI, allowing users to tailor the assistant to any career path, creating a versatile "Dream Job Assistant"
- **Advanced Job Search Tools**: Include an automated job application tracker, enhanced networking features, and guidance on global job opportunities and visas.

### Before starting please install the below packages
"""
logger.info("# GenAI Career Assistant Agent – Your Ultimate Guide to a Career in Generative AI!🚀")

pip install langchain==0.3.7 langchain-community==0.3.7 langchain_google_genai==2.0.4 duckduckgo_search==6.3.4 langgraph==0.2.48 python-dotenv

"""
### Here we will import necessary packages:
`langgraph`, `langchain_core`, `langchain_google_genai` - These are important packages for our project.

This code imports necessary libraries to create and interact with a generative AI model from Google. It loads environment variables to securely set up an API key, then configures the model `gemini-1.5-flash` with specific parameters like verbosity (for detailed logs) and temperature (for response creativity). The AI model is instantiated with the API key to enable its use in generating responses.
"""
logger.info("### Here we will import necessary packages:")



load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             verbose=True,
                             temperature=0.5,
                             google_api_key=os.getenv("GOOGLE_API_KEY"))

"""
### Defining a State class using TypedDict to specify the structure of state data in the workflow.
- `query`: a string representing the user's input or question.
- `category`: a string indicating the category or type of the query.
- `response`: a string holding the AI model's generated response to the query.
This TypedDict ensures each state has a consistent data format for easier management and readability.
"""
logger.info("### Defining a State class using TypedDict to specify the structure of state data in the workflow.")

class State(TypedDict):
    query: str
    category: str
    response: str

"""
### First we are defining utilities we will require further
<a href="https://python.langchain.com/docs/how_to/trim_messages/"> 👉 trim_messages <a>

1. **`trim_conversation` Function**: This function limits the conversation history to the latest messages (up to 10), ensuring only recent and relevant messages are retained in the promp
  
2. **`save_file` Function**: Saves data into a uniquely timestamped Markdown file in the `Agent_output` folder, creating the folder if it doesn't exst.

3. **`show_md_file` Function**: Reads and displays the content of a Markdown file within the notebook, rendering it in Markdown form readabilityblity.
lity.
"""
logger.info("### First we are defining utilities we will require further")


def trim_conversation(prompt):
    """Trims conversation history to retain only the latest messages within the limit."""
    max_messages = 10  # Limit the conversation history to the latest 10 messages
    return trim_messages(
        prompt,
        max_tokens=max_messages,  # Specifies the maximum number of messages allowed
        strategy="last",  # Trimming strategy to keep the last messages
        token_counter=len,  # Counts tokens/messages using the length of the list
        start_on="human",  # Start trimming when reaching the first human message
        include_system=True,  # Include system messages in the trimmed history
        allow_partial=False,  # Ensures only whole messages are included
    )


def save_file(data, filename):
    """Saves data to a markdown file with a timestamped filename."""
    folder_name = "Agent_output"  # Folder to store output files
    os.makedirs(folder_name, exist_ok=True)  # Creates the folder if it doesn't exist

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Format: YYYYMMDDHHMMSS
    filename = f"{filename}_{timestamp}.md"

    file_path = os.path.join(folder_name, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)
        logger.debug(f"File '{file_path}' created successfully.")

    return file_path

def show_md_file(file_path):
    """Displays the content of a markdown file as Markdown in the notebook."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    display(Markdown(content))

"""
#### Now We will create class which will be Responsible for Learning(Tutorial and Q&A sessions)
- Checkout here:
<a href="https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/">👉 create_tool_calling_agent <a>
<a href="https://python.langchain.com/docs/how_to/agent_executor/">👉 AgentExecutor <a>
<a href="https://python.langchain.com/docs/integrations/tools/ddg/">👉 DuckDuckGoSearchResults <a>

1. **Imports**: 
   - `ChatPromptTemplate` and `MessagesPlaceholder` from `langchain_core.prompts` help structure prompts.
   - `DuckDuckGoSearchResults` from `langchain_community.tools` provides web search capability.
   - `create_tool_calling_agent` and `AgentExecutor` manage agent creation and execution.

2. **`LearningResourceAgent` Class**:
   - **`__init__` Method**: Initializes the chat model (`gemini-1.5-pro`), prompt, and tools (like DuckDuckGo search).
   - **`TutorialAgent` Method**: Runs a search-based tutorial by invoking the model and saving the output as a timestamped Markdown file for review.
   - **`QueryBot` Method**: Conducts a Q&A loop with the user. The conversation is trimmed as it grows, and responses are generated based on updated inputs, with the user able to type 'exit' to end the session.
"""
logger.info("#### Now We will create class which will be Responsible for Learning(Tutorial and Q&A sessions)")


class LearningResourceAgent:
    def __init__(self, prompt):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def TutorialAgent(self, user_input):
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        response = agent_executor.invoke({"input": user_input})

        path = save_file(str(response.get('output')).replace("```markdown", "").strip(), 'Tutorial')
        logger.debug(f"Tutorial saved to {path}")
        return path

    def QueryBot(self, user_input):
        logger.debug("\nStarting the Q&A session. Type 'exit' to end the session.\n")
        record_QA_session = []
        record_QA_session.append('User Query: %s \n' % user_input)
        self.prompt.append(HumanMessage(content=user_input))
        while True:
            self.prompt = trim_conversation(self.prompt)

            response = self.model.invoke(self.prompt)
            record_QA_session.append('\nExpert Response: %s \n' % response.content)

            self.prompt.append(AIMessage(content=response.content))

            logger.debug('*' * 50 + 'AGENT' + '*' * 50)
            logger.debug("\nEXPERT AGENT RESPONSE:", response.content)

            logger.debug('*' * 50 + 'USER' + '*' * 50)
            user_input = input("\nYOUR QUERY: ")
            record_QA_session.append('\nUser Query: %s \n' % response.content)
            self.prompt.append(HumanMessage(content=user_input))

            if user_input.lower() == "exit":
                logger.debug("Ending the chat session.")
                path = save_file(''.join(record_QA_session),'Q&A_Doubt_Session')
                logger.debug(f"Q&A Session saved to {path}")
                return path

"""
### Here we are creating Class for Interview handling(Interview Question Prep and MockInterview)

1. **`InterviewAgent` Class**:
   - **`__init__` Method**: Initializes the model (`gemini-1.5-flash`), prompt, tools (such as DuckDuckGo search), and creates an agent executor with error handling enabled.

2. **`Interview_questions` Method**:
   - Runs a loop for handling interview questions from the user, generating responses using the agent executor. 
   - Responses are stored in `questions_bank` for later reference. The conversation ends when the user types "exit," and the chat history is saved as a Markdown file.

3. **`Mock_Interview` Method**:
   - Simulates a mock interview by initiating a conversation loop. Responses from the model are displayed as "Interviewer" messages, and user inputs as "Candidate" messages.
   - The conversation is trimmed as it grows to maintain prompt size. The interview session ends when the user types "exit," returning a record of the entire interview.
"""
logger.info("### Here we are creating Class for Interview handling(Interview Question Prep and MockInterview)")

class InterviewAgent:
    def __init__(self, prompt):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]  # Web search tool for retrieving additional information

    def Interview_questions(self, user_input):
        chat_history = []
        questions_bank = ''
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        while True:
            logger.debug("\nStarting the Interview question preparation. Type 'exit' to end the session.\n")
            if user_input.lower() == "exit":
                logger.debug("Ending the conversation. Goodbye!")
                break

            response = self.agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            questions_bank += str(response.get('output')).replace("```markdown", "").strip() + "\n"

            chat_history.extend([HumanMessage(content=user_input), response["output"]])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]  # Keep only the last 10 messages

            user_input = input("You: ")

        path = save_file(questions_bank, 'Interview_questions')
        logger.debug(f"Interviews question saved to {path}")
        return path

    def Mock_Interview(self):
        logger.debug("\nStarting the mock interview. Type 'exit' to end the session.\n")

        initial_message = 'I am ready for the interview.\n'
        interview_record = []
        interview_record.append('Candidate: %s \n' % initial_message)
        self.prompt.append(HumanMessage(content=initial_message))

        while True:
            self.prompt = trim_conversation(self.prompt)

            response = self.model.invoke(self.prompt)

            self.prompt.append(AIMessage(content=response.content))

            logger.debug("\nInterviewer:", response.content)
            interview_record.append('\nInterviewer: %s \n' % response.content)

            user_input = input("\nCandidate: ")
            interview_record.append('\nCandidate: %s \n' % user_input)

            self.prompt.append(HumanMessage(content=user_input))

            if user_input.lower() == "exit":
                logger.debug("Ending the interview session.")
                path = save_file(''.join(interview_record),'Mock_Interview')
                logger.debug(f"Mock Interview saved to {path}")
                return path

"""
### Now we will create class for Resume Making which will handle creating resume by chating with user

1. **`ResumeMaker` Class**:
   - **`__init__` Method**: Initializes the chat model (`gemini-1.5-pro`), prompt, and tools (like DuckDuckGo search). Sets up an agent executor with tool access, verbose output, and error handling for generating resume content.

2. **`Create_Resume` Method**:
   - Engages in a conversational loop to gather user input and generate resume content based on responses from the agent.
   - The conversation history (`chat_history`) retains only the latest 10 messages to manage context size.
   - The conversation loop ends when the user types "exit," at which point the final output is saved as a timestamped Markdown file titled "Resume," and the file path is displayed.
"""
logger.info("### Now we will create class for Resume Making which will handle creating resume by chating with user")

class ResumeMaker:
    def __init__(self, prompt):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]  # Search tool to gather additional information if needed
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    def Create_Resume(self, user_input):
        chat_history = []
        while True:
            logger.debug("\nStarting the Resume create session. Type 'exit' to end the session.\n")
            if user_input.lower() == "exit":
                logger.debug("Ending the conversation. Goodbye!")
                break

            response = self.agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            chat_history.extend([HumanMessage(content=user_input), response["output"]])

            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

            user_input = input("You: ")

        path = save_file(str(response.get('output')).replace("```markdown", "").strip(), 'Resume')
        logger.debug(f"Resume saved to {path}")
        return path

"""
### Code Explanation

1. **`JobSearch` Class**:
   - **`__init__` Method**: Initializes the chat model (`gemini-1.5-pro`), prompt, and search tools for job search assistance. Sets up an agent executor to handle conversation flow with verbose output and error handling.

2. **`find_jobs` Method**:
   - Conducts a conversational loop to assist users with job search queries, using the agent's responses based on user input.
   - Retains only the latest 10 messages in `chat_history` to manage the prompt size effectively.
   - The loop ends when the user types "exit," after which the final conversation output is saved to a Markdown file titled "Resume," and the file path is displayed to the user.
"""
logger.info("### Code Explanation")

class JobSearch:
    def __init__(self, prompt):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.prompt = prompt
        self.tools = DuckDuckGoSearchResults()  # Search tool to find job listings or related information

    def find_jobs(self, user_input):
        results = self.tools.invoke(user_input)
        chain = self.prompt | self.model
        jobs = chain.invoke({"result": results}).content

        path = save_file(str(jobs).replace("```markdown", "").strip(), 'Job_search')
        logger.debug(f"Jobs saved to {path}")
        return path

"""
### Now we are creating function which will help to categorize our user Input
- We are using <a href="https://python.langchain.com/docs/how_to/few_shot_examples/">👉 Few Shot prompting </a>(Checkout here) to make our LLM understand the categories.
1. **`categorize` Function**:
   - Categorizes a user query into four main areas: Learning Generative AI, Resume Making, Interview Preparation, or Job Search.
   - Uses a template prompt with examples to guide the AI in choosing the correct category and returns the category number.

2. **`handle_learning_resource` Function**:
   - Determines if a user query about generative AI is related to creating tutorials or asking general questions.
   - Uses a prompt to specify these categories and returns "Tutorial" or "Question" based on the AI's categorization.

3. **`handle_interview_preparation` Function**:
   - Identifies if a user query in the interview category is about a mock interview or general interview questions.
   - Uses examples to instruct the AI on the difference, returning either "Mock" or "Question" to guide further interaction.
"""
logger.info("### Now we are creating function which will help to categorize our user Input")

def categorize(state: State) -> State:
    """Categorizes the user query into one of four main categories: Learn Generative AI Technology, Resume Making, Interview Preparation, or Job Search."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories:\n"
        "1: Learn Generative AI Technology\n"
        "2: Resume Making\n"
        "3: Interview Preparation\n"
        "4: Job Search\n"
        "Give the number only as an output.\n\n"
        "Examples:\n"
        "1. Query: 'What are the basics of generative AI, and how can I start learning it?' -> 1\n"
        "2. Query: 'Can you help me improve my resume for a tech position?' -> 2\n"
        "3. Query: 'What are some common questions asked in AI interviews?' -> 3\n"
        "4. Query: 'Are there any job openings for AI engineers?' -> 4\n\n"
        "Now, categorize the following customer query:\n"
        "Query: {query}"
    )

    chain = prompt | llm
    logger.debug('Categorizing the customer query...')
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

def handle_learning_resource(state: State) -> State:
    """Determines if the query is related to Tutorial creation or general Questions on generative AI topics."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these categories:\n\n"
        "Categories:\n"
        "- Tutorial: For queries related to creating tutorials, blogs, or documentation on generative AI.\n"
        "- Question: For general queries asking about generative AI topics.\n"
        "- Default to Question if the query doesn't fit either of these categories.\n\n"
        "Examples:\n"
        "1. User query: 'How to create a blog on prompt engineering for generative AI?' -> Category: Tutorial\n"
        "2. User query: 'Can you provide a step-by-step guide on fine-tuning a generative model?' -> Category: Tutorial\n"
        "3. User query: 'Provide me the documentation for Langchain?' -> Category: Tutorial\n"
        "4. User query: 'What are the main applications of generative AI?' -> Category: Question\n"
        "5. User query: 'Is there any generative AI course available?' -> Category: Question\n\n"
        "Now, categorize the following user query:\n"
        "The user query is: {query}\n"
    )

    chain = prompt | llm
    logger.debug('Categorizing the customer query further...')
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}

def handle_interview_preparation(state: State) -> State:
    """Determines if the query is related to Mock Interviews or general Interview Questions."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these categories:\n\n"
        "Categories:\n"
        "- Mock: For requests related to mock interviews.\n"
        "- Question: For general queries asking about interview topics or preparation.\n"
        "- Default to Question if the query doesn't fit either of these categories.\n\n"
        "Examples:\n"
        "1. User query: 'Can you conduct a mock interview with me for a Gen AI role?' -> Category: Mock\n"
        "2. User query: 'What topics should I prepare for an AI Engineer interview?' -> Category: Question\n"
        "3. User query: 'I need to practice interview focused on Gen AI.' -> Category: Mock\n"
        "4. User query: 'Can you list important coding topics for AI tech interviews?' -> Category: Question\n\n"
        "Now, categorize the following user query:\n"
        "The user query is: {query}\n"
    )

    chain = prompt | llm
    logger.debug('Categorizing the customer query further...')
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}

"""
### Now we will create function for job search and Resume making

1. **`job_search` Function**:
   - Sets up a job search agent to find Generative AI job listings based on user input, gathering details like company name, job title, and links.
   - Generates a Markdown (.md) file with results, displayed to the user.

2. **`handle_resume_making` Function**:
   - Creates a customized resume for AI-focused roles by gathering user details (skills, experience, projects) in a structured format.
   - Produces a Markdown (.md) resume template tailored to the Generative AI jobcandidate's performance.
"""
logger.info("### Now we will create function for job search and Resume making")

def job_search(state: State) -> State:
    """Provide a job search response based on user query requirements."""
    prompt = ChatPromptTemplate.from_template('''Your task is to refactor and make .md file for the this content which includes
    the jobs available in the market. Refactor such that user can refer easily. Content: {result}''')
    jobSearch = JobSearch(prompt)
    state["query"] = input('Please make sure to mention Job location you want,Job roles\n')
    path = jobSearch.find_jobs(state["query"])
    show_md_file(path)
    return {"response": path}

def handle_resume_making(state: State) -> State:
    """Generate a customized resume based on user details for a tech role in AI and Generative AI."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are a skilled resume expert with extensive experience in crafting resumes tailored for tech roles, especially in AI and Generative AI.
        Your task is to create a resume template for an AI Engineer specializing in Generative AI, incorporating trending keywords and technologies in the current job market.
        Feel free to ask users for any necessary details such as skills, experience, or projects to complete the resume.
        Try to ask details step by step and try to ask all details within 4 to 5 steps.
        Ensure the final resume is in .md format.'''),
       MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    resumeMaker = ResumeMaker(prompt)
    path = resumeMaker.Create_Resume(state["query"])
    show_md_file(path)
    return {"response": path}

"""
### Next we will create a function for Q&A query bot and Tutorial maker

3. **`ask_query_bot` Function**:
   - Engages in a conversational Q&A session, providing detailed answers to user queries related to Generative AI.
   - Uses back-and-forth interaction to ensure clarity and completeness in responses.

4. **`tutorial_agent` Function**:
   - Generates comprehensive tutorial blogs on Generative AI topics with explanations, example code, and resources for further learning.
   - Saves the tutorial in Markdown (.md) format, designed for clarity and learning support.
"""
logger.info("### Next we will create a function for Q&A query bot and Tutorial maker")

def ask_query_bot(state: State) -> State:
    """Provide detailed answers to user queries related to Generative AI."""
    system_message = '''You are an expert Generative AI Engineer with extensive experience in training and guiding others in AI engineering.
    You have a strong track record of solving complex problems and addressing various challenges in AI.
    Your role is to assist users by providing insightful solutions and expert advice on their queries.
    Engage in a back-and-forth chat session to address user queries.'''
    prompt = [SystemMessage(content=system_message)]

    learning_agent = LearningResourceAgent(prompt)

    path = learning_agent.QueryBot(state["query"])
    show_md_file(path)
    return {"response": path}

def tutorial_agent(state: State) -> State:
    """Generate a tutorial blog for Generative AI based on user requirements."""
    system_message = '''You are a knowledgeable assistant specializing as a Senior Generative AI Developer with extensive experience in both development and tutoring.
         Additionally, you are an experienced blogger who creates tutorials focused on Generative AI.
         Your task is to develop high-quality tutorials blogs in .md file with Coding example based on the user's requirements.
         Ensure tutorial includes clear explanations, well-structured python code, comments, and fully functional code examples.
         Provide resource reference links at the end of each tutorial for further learning.'''
    prompt = ChatPromptTemplate.from_messages([("system", system_message),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),])
    learning_agent = LearningResourceAgent(prompt)
    path = learning_agent.TutorialAgent(state["query"])
    show_md_file(path)
    return {"response": path}

"""
### Finally we will create function Interview Question prep and Mock interview

5. **`interview_topics_quesions` Function**:
   - Provides a list of interview questions tailored to Generative AI roles, along with references where possible.
   - Outputs a Markdown (.md) document with curated questions based on user requirements.

6. **`mock_interview` Function**:
   - Conducts a simulated Generative AI job interview, interacting with the user in real-time.
   - Provides a post-interview evaluation, summarizing the candidate's performance.
"""
logger.info("### Finally we will create function Interview Question prep and Mock interview")

def interview_topics_questions(state: State) -> State:
    """Provide a curated list of interview questions related to Generative AI based on user input."""
    system_message = '''You are a good researcher in finding interview questions for Generative AI topics and jobs.
                     Your task is to provide a list of interview questions for Generative AI topics and job based on user requirements.
                     Provide top questions with references and links if possible. You may ask for clarification if needed.
                     Generate a .md document containing the questions.'''
    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_message),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                        ("placeholder", "{agent_scratchpad}"),])
    interview_agent = InterviewAgent(prompt)
    path = interview_agent.Interview_questions(state["query"])
    show_md_file(path)
    return {"response": path}

def mock_interview(state: State) -> State:
    """Conduct a mock interview for a Generative AI position, including evaluation at the end."""
    system_message = '''You are a Generative AI Interviewer. You have conducted numerous interviews for Generative AI roles.
         Your task is to conduct a mock interview for a Generative AI position, engaging in a back-and-forth interview session.
         The conversation should not exceed more than 15 to 20 minutes.
         At the end of the interview, provide an evaluation for the candidate.'''
    prompt = [SystemMessage(content=system_message)]
    interview_agent = InterviewAgent(prompt)
    path = interview_agent.Mock_Interview()
    show_md_file(path)
    return {"response": path}

"""
### Here, We are creating routing function which will be responsible for conditional edge to give direction after categorization.
- Checkout Here<a href="https://langchain-ai.github.io/langgraph/#:~:text=Conditional%20edge%20means%20that%20the,the%20agent%20(LLM)%20decides.">👉 Conditional Edge</a>
1. **`route_query` Function**:
   - Routes the main query based on the assigned category number, directing it to the appropriate handler: Learning Resource, Resume Making, Interview Preparation, or Job Search.
   - If the query does not match any predefined categories, prompts the user to ask a more relevant question.

2. **`route_interview` Function**:
   - Routes interview-related queries to specific handlers based on the query's sub-category (either Mock Interview or Interview Topics).
   - If the category is unclear, defaults to "mock_interview."

3. **`route_learning` Function**:
   - Routes learning-related queries, directing them to either a general question bot or a tutorial creation agent.
   - Returns `False` if the query does not clearly fit either sub-category.
"""
logger.info("### Here, We are creating routing function which will be responsible for conditional edge to give direction after categorization.")

def route_query(state: State):
    """Route the query based on its category to the appropriate handler."""
    if '1' in state["category"]:
        logger.debug('Category: handle_learning_resource')
        return "handle_learning_resource"  # Directs queries about learning generative AI to the learning resource handler
    elif '2' in state["category"]:
        logger.debug('Category: handle_resume_making')
        return "handle_resume_making"  # Directs queries about resume making to the resume handler
    elif '3' in state["category"]:
        logger.debug('Category: handle_interview_preparation')
        return "handle_interview_preparation"  # Directs queries about interview preparation to the interview handler
    elif '4' in state["category"]:
        logger.debug('Category: job_search')
        return "job_search"  # Directs job search queries to the job search handler
    else:
        logger.debug("Please ask your question based on my description.")
        return False  # Returns False if the category does not match any predefined options

def route_interview(state: State) -> str:
    """Route the query to the appropriate interview-related handler."""
    if 'Question'.lower() in state["category"].lower():
        logger.debug('Category: interview_topics_questions')
        return "interview_topics_questions"  # Directs to the handler for interview topic questions
    elif 'Mock'.lower() in state["category"].lower():
        logger.debug('Category: mock_interview')
        return "mock_interview"  # Directs to the mock interview handler
    else:
        logger.debug('Category: mock_interview')
        return "mock_interview"  # Defaults to mock interview if category does not clearly match

def route_learning(state: State):
    """Route the query based on the learning path category."""
    if 'Question'.lower() in state["category"].lower():
        logger.debug('Category: ask_query_bot')
        return "ask_query_bot"  # Directs queries to the general question bot
    elif 'Tutorial'.lower() in state["category"].lower():
        logger.debug('Category: tutorial_agent')
        return "tutorial_agent"  # Directs queries to the tutorial creation agent
    else:
        logger.debug("Please ask your question based on my interview description.")
        return False  # Returns False if no clear category match is found

"""
### Now all set lets create workflow graphs adding edges and nodes.

1. **Workflow Creation**:
   - Initializes a `StateGraph` to define the workflow for query handling, with each node representing a different query handler.

2. **Nodes and Edges**:
   - Adds nodes for each query category and handling function: `categorize`, `handle_learning_resource`, `handle_resume_making`, `handle_interview_preparation`, and `job_search`.
   - Adds edges to connect nodes conditionally based on the category (e.g., `route_query` routes from "categorize" to the appropriate handler).
   - Conditional edges further route within specific categories, such as learning resources (`route_learning`) and interview preparation (`route_interview`).

3. **Workflow Endpoints**:
   - Defines nodes where the workflow terminates (e.g., `handle_resume_making`, `mock_interview`, and `job_search`), connecting them to `END`.

4. **Compilation**:
   - Sets `categorize` as the entry point and compiles the workflow into an executable application, `app`, for handling user queries.
"""
logger.info("### Now all set lets create workflow graphs adding edges and nodes.")

workflow = StateGraph(State)

workflow.add_node("categorize", categorize)  # Initial categorization node
workflow.add_node("handle_learning_resource", handle_learning_resource)  # Handles learning-related queries
workflow.add_node("handle_resume_making", handle_resume_making)  # Handles resume-making queries
workflow.add_node("handle_interview_preparation", handle_interview_preparation)  # Handles interview prep queries
workflow.add_node("job_search", job_search)  # Handles job search queries
workflow.add_node("mock_interview", mock_interview)  # Handles mock interview sessions
workflow.add_node("interview_topics_questions", interview_topics_questions)  # Handles interview topic questions
workflow.add_node("tutorial_agent", tutorial_agent)  # Tutorial agent for generative AI learning resources
workflow.add_node("ask_query_bot", ask_query_bot)  # General query bot for learning resources

workflow.add_edge(START, "categorize")

workflow.add_conditional_edges(
    "categorize",
    route_query,
    {
        "handle_learning_resource": "handle_learning_resource",
        "handle_resume_making": "handle_resume_making",
        "handle_interview_preparation": "handle_interview_preparation",
        "job_search": "job_search"
    }
)

workflow.add_conditional_edges(
    "handle_interview_preparation",
    route_interview,
    {
        "mock_interview": "mock_interview",
        "interview_topics_questions": "interview_topics_questions",
    }
)

workflow.add_conditional_edges(
    "handle_learning_resource",
    route_learning,
    {
        "tutorial_agent": "tutorial_agent",
        "ask_query_bot": "ask_query_bot",
    }
)

workflow.add_edge("handle_resume_making", END)
workflow.add_edge("job_search", END)
workflow.add_edge("interview_topics_questions", END)
workflow.add_edge("mock_interview", END)
workflow.add_edge("ask_query_bot", END)
workflow.add_edge("tutorial_agent", END)

workflow.set_entry_point("categorize")

app = workflow.compile()

"""
### Lets Visualize our graph

- **Displaying Workflow Graph**:
  - Generates a visual representation of the `app` workflow graph using Mermaid, which is then displayed as a PNG image.
  - The `MermaidDrawMethod.API` method is used to create the PNG, ensuring a clear, structured view of the workflow nodes and their connections.
"""
logger.info("### Lets Visualize our graph")

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,  # Uses Mermaid's API to generate the PNG image of the workflow graph
        )
    )
)

"""
### Final function to Processes a user query using the LangGraph workflow and returns a dictionary containing the query's category and response.
"""
logger.info("### Final function to Processes a user query using the LangGraph workflow and returns a dictionary containing the query's category and response.")

def run_user_query(query: str) -> Dict[str, str]:
    """Process a user query through the LangGraph workflow.

    Args:
        query (str): The user's query

    Returns:
        Dict[str, str]: A dictionary containing the query's category and response
    """
    results = app.invoke({"query": query})
    return {
        "category": results["category"],
        "response": results["response"]
    }

"""
# ---------------------------Testing Different Scenarios------------------------------

## TEST CASE 1: Creating Tutorials
"""
logger.info("# ---------------------------Testing Different Scenarios------------------------------")

query = "I want to learn Langchain and langgraph.With usage and concept. Also give coding example implementation for both.Create tutorial for this."
result = run_user_query(query)
result

"""
## TEST CASE 2: Q&A session for Doubts
"""
logger.info("## TEST CASE 2: Q&A session for Doubts")

query = "I am confused between Langgraph and CrewAI when to use what for Agent Creation?"
result = run_user_query(query)
logger.debug(result)

"""
## TEST CASE 3: Interview Question Discussion
"""
logger.info("## TEST CASE 3: Interview Question Discussion")

query = "I want to discussion Interview question for Gen AI job roles."
result = run_user_query(query)
logger.debug(result)

"""
## TEST CASE 4: Mock Interview with Evaluation Feedback
"""
logger.info("## TEST CASE 4: Mock Interview with Evaluation Feedback")

query = "I need mock interview to practice."
result = run_user_query(query)
result

"""
## TEST CASE 5: Resume Modification Based on Job Description
"""
logger.info("## TEST CASE 5: Resume Modification Based on Job Description")

query = "Can you help me to modify my resume based on job description"
result = run_user_query(query)
result

"""
## TEST CASE 6: Resume Making
"""
logger.info("## TEST CASE 6: Resume Making")

query = "I want to make resume for Gen AI roles job."
result = run_user_query(query)
result

"""
## TEST CASE 7: Job Search
"""
logger.info("## TEST CASE 7: Job Search")

query = "I want to search jobs."

result = run_user_query(query)
result

logger.info("\n\n[DONE]", bright=True)