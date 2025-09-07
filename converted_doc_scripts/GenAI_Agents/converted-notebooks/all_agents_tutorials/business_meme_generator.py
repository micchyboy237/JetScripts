async def main():
    from jet.transformers.formatters import format_json
    from IPython.display import display, Image as IPImage
    from PIL import Image
    from dotenv import load_dotenv
    from groq import Groq
    from io import BytesIO
    from jet.logger import CustomLogger
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.runnables.graph import MermaidDrawMethod
    from langchain_groq import ChatGroq
    from langgraph.graph import Graph, END
    from pydantic import BaseModel, Field
    from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
    from urllib.parse import quote
    import aiohttp
    import asyncio
    import io
    import json
    import os
    import random
    import re
    import requests
    import shutil
    import traceback
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # Business Meme Generator using LangGraph and Memegen.link
    
    ## Overview
    This project demonstrates the creation of a business meme generator that leverages LLMs and the Memegen.link API. By combining LangGraph for workflow management, Groq with `llama-3.1-70b-versatile` for text generation and company information extraction, and Memegen.link for meme creation, we've developed a system that can produce contextually relevant memes based on company website analysis.
    
    ## Motivation
    In the modern digital marketing landscape, memes have become a powerful tool for brand communication and engagement. This project aims to showcase how AI technologies can be integrated to create a workflow that analyzes a company's online presence and automatically generates relevant, brand-aligned memes. This tool could be valuable for digital marketers, social media managers, and brand strategists looking to create engaging content efficiently.
    
    ## Key Components
    1. **LangGraph**: Orchestrates the overall workflow, managing the flow of data between different stages of the process.
    2. **Llama 3.1 70b (via Groq)**: Analyzes website content and generates meme concepts and text based on company context.
    3. **Memegen.link API**: Provides meme templates and handles meme image generation.
    4. **Pydantic Models**: Ensures type safety and data validation throughout the workflow.
    5. **Asynchronous Programming**: Utilizes `asyncio` and `aiohttp` for efficient parallel processing.
    
    ## Method
    The meme generation process follows these high-level steps:
    
    1. **Website Analysis**:
       - Fetches and analyzes the company's website content
       - Extracts key information about brand tone, target audience, and value proposition
    
    2. **Context Generation**:
       - Creates a structured company context including:
         - Brand tone of voice
         - Target audience
         - Value proposition
         - Key products/services
         - Brand personality traits
    
    3. **Meme Concept Creation**:
       - Generates multiple meme concepts based on the company context
       - Each concept includes:
         - Main message/joke
         - Intended emotional response
         - Audience relevance
    
    4. **Template Selection**:
       - Fetches available meme templates from Memegen.link
       - Matches concepts with appropriate templates based on context
    
    5. **Text Generation**:
       - Creates contextually appropriate text for each meme
       - Ensures alignment with brand voice and message
    
    6. **Meme Assembly**:
       - Combines selected templates with generated text
       - Creates final meme URLs using Memegen.link API
    
    ## Data Structures
    The project uses several key Pydantic models for data validation:
    
    1. **CompanyContext**:
       - Structured representation of company information
       - Includes tone, target audience, value proposition, etc.
    
    2. **MemeConcept**:
       - Represents individual meme ideas
       - Contains message, emotion, and audience relevance
    
    3. **TemplateInfo**:
       - Stores meme template metadata
       - Includes template ID, name, description, and example text
    
    4. **GeneratedMeme**:
       - Final meme output structure
       - Contains all template and text information plus final URL
    
    ## Workflow Components
    The LangGraph workflow consists of several key nodes:
    
    1. `get_website_content`:
       - Fetches and processes website content
       - Handles URL validation and content extraction
    
    2. `analyze_company_insights`:
       - Processes website content into structured company context
       - Uses LLM for content analysis
    
    3. `generate_meme_concepts`:
       - Creates meme concepts based on company context
       - Ensures brand alignment
    
    4. `select_meme_templates`:
       - Matches concepts with appropriate templates
       - Handles template filtering and selection
    
    5. `generate_text_elements`:
       - Creates meme text based on concepts and templates
       - Maintains brand voice consistency
    
    6. `create_meme_url`:
       - Generates final meme URLs
       - Handles URL encoding and formatting
    
    ## Usage
    The system can be used by providing a company website URL:
    
    ```python
    website_url = "https://www.langchain.com"
    result = await run_workflow(website_url)
    ```
    
    The workflow will analyze the website, generate appropriate memes, and display:
    - Company context analysis
    - Generated meme concepts
    - Final memes with captions
    - Meme preview images
    
    ## Conclusion
    This Business Meme Generator demonstrates the potential of combining different technologies with AI to create a powerful content generation tool. The modular nature of the system, facilitated by LangGraph, allows for easy updates or replacements of individual components as technologies evolve.
    
    The project shows how AI can assist in creative tasks while maintaining brand consistency and relevance. Future enhancements could include:
    - Additional template sources
    - More sophisticated brand analysis
    - Integration with social media accounts to better understand brand voice
    - User feedback integration
    - Custom template upload capabilities
    
    As AI continues to evolve, tools like this will become increasingly valuable for digital marketing and brand communication strategies.
    
    ## Installing Libraries
    Intall necessary libraries.
    """
    logger.info("# Business Meme Generator using LangGraph and Memegen.link")
    
    # !pip install langgraph langchain_groq langchain_core IPython python-dotenv groq langchain_community
    
    """
    ## Import Dependencies
    
    This cell imports all necessary libraries and sets up the environment.
    """
    logger.info("## Import Dependencies")
    
    
    
    
    load_dotenv()
    
    os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
    
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    """
    ## Define Data Structures and GraphState
    
    Define the data structure template models for validation using Pydantic models and graph state using TypedDict.
    """
    logger.info("## Define Data Structures and GraphState")
    
    class CompanyContext(BaseModel):
        """Company context to be used to generate memes."""
        tone: str = Field(description = "The tone of voice of the company")
        target_audience: str = Field(description = "The target audience of the company")
        value_proposition: str = Field(description = "The value proposition of the company")
        key_products: List[str] = Field(description = "A list with the company's key products")
        brand_personality: str = Field(description = "The brand personality of the company")
    
    class MemeConcept(BaseModel):
        message: str = Field(description="The core message of the meme")
        emotion: str = Field(description="The emotion conveyed by the meme")
        audience_relevance: str = Field(description="The relevance of the meme to the audience")
    
    class MemeConcepts(BaseModel):
        concepts: List[MemeConcept] = Field(description="List of meme concepts")
    
    class TemplateInfo(BaseModel):
        template_id: str = Field(..., description="Unique identifier for the template")
        name: str = Field(..., description="Name of the meme template")
        blank_template_api_link: str = Field(..., description="API link to the blank template")
        description: str = Field(..., description="Description of the meme template")
        example_text_1: Optional[str] = Field("", description="Example text for the first line")
        example_text_2: Optional[str] = Field("", description="Example text for the second line")
        lines: int = Field(..., description="Number of text lines in the meme")
        keywords: List[str] = Field(..., description="Keywords associated with the template")
    
    class SelectedMeme(BaseModel):
        meme_id: str = Field(..., description="Unique identifier for the selected meme")
        template_id: str = Field(..., description="ID of the selected template")
        concept: MemeConcept = Field(..., description="The concept associated with the meme")
        is_text_element1_filled: bool = Field(..., description="Indicates if the first text element is filled")
        is_text_element2_filled: bool = Field(..., description="Indicates if the second text element is filled")
        example_text_1: Optional[str] = Field("", description="Example text for the first line")
        example_text_2: Optional[str] = Field("", description="Example text for the second line")
        template_info: TemplateInfo = Field(..., description="Information about the selected template")
        blank_template_api_link: str = Field(..., description="API link to the blank template without extension")
        blank_template_api_link_extension: str = Field(..., description="File extension of the blank template link")
    
    class PreGeneratedMeme(BaseModel):
        meme_id: str = Field(..., description="Unique identifier for the selected meme")
        template_id: str = Field(..., description="ID of the selected template")
        concept: MemeConcept = Field(..., description="The concept associated with the meme")
        is_text_element1_filled: bool = Field(..., description="Indicates if the first text element is filled")
        is_text_element2_filled: bool = Field(..., description="Indicates if the second text element is filled")
        template_info: TemplateInfo = Field(..., description="Information about the selected template")
        blank_template_api_link: str = Field(..., description="API link to the blank template without extension")
        blank_template_api_link_extension: str = Field(..., description="File extension of the blank template link")
        generated_text_element1: str = Field(..., description="Generated text element 1")
        generated_text_element2: str = Field(..., description="Generated text element 2")
    
    class GeneratedMeme(BaseModel):
        meme_id: str = Field(..., description="Unique identifier for the selected meme")
        template_id: str = Field(..., description="ID of the selected template")
        concept: MemeConcept = Field(..., description="The concept associated with the meme")
        is_text_element1_filled: bool = Field(..., description="Indicates if the first text element is filled")
        is_text_element2_filled: bool = Field(..., description="Indicates if the second text element is filled")
        template_info: TemplateInfo = Field(..., description="Information about the selected template")
        blank_template_api_link: str = Field(..., description="API link to the blank template without extension")
        blank_template_api_link_extension: str = Field(..., description="File extension of the blank template link")
        generated_text_element1: str = Field(..., description="Generated text element 1")
        generated_text_element2: str = Field(..., description="Generated text element 2")
        treated_text_element1: str = Field(..., description="Treated text element 1")
        treated_text_element2: str = Field(..., description="Treated text element 2")
        final_url: str = Field(..., description="Final URL of the generated meme")
    
    class GraphState(TypedDict):
        """Enhanced state object for the meme generation workflow."""
        messages: Annotated[Sequence[HumanMessage | AIMessage], "Conversation messages"]
        website_url: Annotated[str, "Company website URL"]
        website_content: Annotated[List, "Website content"]
        company_context: Annotated[Dict[str, Any], "Analyzed company information"]
        meme_concepts: Annotated[List[dict], "Generated meme concepts"]
        selected_concepts: Annotated[List[dict], "Selected top 3 meme concepts"]
        selected_memes: Annotated[Dict[str, SelectedMeme], "Selected memes with their info"]
        pre_generated_memes: Annotated[Dict[str, PreGeneratedMeme], "Pre-Generated memes with their info"]
        generated_memes: Annotated[Dict[str, GeneratedMeme], "Pre-Generated memes with their info"]
        available_templates: Annotated[Dict[str, TemplateInfo], "Available meme templates"]
    
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    """
    ## Define Graph Functions
    
    Define the functions that will be used in the LangGraph workflow.
    """
    logger.info("## Define Graph Functions")
    
    def ensure_url(string: str) -> str:
        """
        Ensures a given string is a properly formatted URL by adding 'http://' if needed
        and validating the URL format.
    
        Args:
            string (str): The URL string to validate and format
    
        Returns:
            str: A properly formatted URL
    
        Raises:
            ValueError: If the URL format is invalid
    
        Example:
            >>> ensure_url("example.com")
            'http://example.com'
            >>> ensure_url("https://example.com")
            'https://example.com'
        """
    
        if not string.startswith(("http://", "https://")):
            string = "http://" + string
    
        url_regex = re.compile(
            r"^(https?:\/\/)?"  # optional protocol
            r"(www\.)?"  # optional www
            r"([a-zA-Z0-9.-]+)"  # domain
            r"(\.[a-zA-Z]{2,})?"  # top-level domain
            r"(:\d+)?"  # optional port
            r"(\/[^\s]*)?$",  # optional path
            re.IGNORECASE,
        )
    
        if not url_regex.match(string):
            msg = f"Invalid URL: {string}"
            raise ValueError(msg)
    
        return string
    
    async def get_website_content(state: GraphState) -> GraphState:
        """
        Fetches and analyzes website content using WebBaseLoader.
    
        Args:
            state (GraphState): Current workflow state containing website_url
    
        Returns:
            GraphState: Updated state with website_content added
    
        Notes:
            - Uses WebBaseLoader to fetch HTML content
            - Handles encoding with utf-8
            - Updates state with combined text content from all pages
            - Handles errors and updates state with error message if fetch fails
        """
    
        try:
            website_url = state["website_url"]
            validated_url = ensure_url(website_url)
    
            web_loader = WebBaseLoader(web_paths=[validated_url], encoding="utf-8")
            text_docs = web_loader.load()
    
            content = []
            for doc in text_docs:
                content.append(doc.page_content)
    
            state["website_content"] = "\n\n".join(content)
            return state
    
        except Exception as e:
            logger.debug(f"Error fetching website content: {str(e)}")
            state["website_content"] = f"Error fetching content from {website_url}: {str(e)}"
            return state
    
    def analyze_company_insights(state: GraphState) -> GraphState:
        """
        Analyzes company information from website content using LLM.
    
        Args:
            state (GraphState): Current state containing website_content
    
        Returns:
            GraphState: Updated state with company_context added
    
        Notes:
            - Uses structured LLM output for consistent format
            - Analyzes tone, target audience, value proposition, key products, and brand personality
            - Handles cases where no content is available
        """
    
        website_data = state.get("website_content")
        if not website_data:
            state["company_context"] = {"error": "No content available to analyze."}
            return state
    
        content = website_data[0]
    
        prompt = f"""Analyze this company website content and provide insights in a JSON format with the following structure:
        {{
            "tone": "string describing the brand tone of voice (professional, casual, technical, etc.)",
            "target_audience": "string describing target audience/persona",
            "value_proposition": "string describing their unique value proposition",
            "key_products": ["array", "of", "key", "products", "or", "services"],
            "brand_personality": "string describing 3-5 key brand personality traits"
        }}
    
        Website Content:
        {website_data}
    
        Please ensure key_products is always returned as an array/list, even if there's only one product.
        Be specific and base insights on the actual content."""
    
        structured_llm = llm.with_structured_output(CompanyContext)
    
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        state["company_context"] = response
    
        return state
    
    def generate_meme_concepts(state: GraphState) -> GraphState:
        """
        Generates meme concepts based on analyzed company insights.
    
        Args:
            state (GraphState): Current state containing company_context
    
        Returns:
            GraphState: Updated state with meme_concepts and selected_concepts added
    
        Notes:
            - Creates 3 meme concepts based on company insights
            - Each concept includes message, emotion, and audience relevance
            - Parses JSON response to extract structured concepts
            - Selects top 3 concepts for further processing
        """
    
        insights = state["company_context"]
    
        prompt = f"""Create 3 meme concepts based on these company insights:
    
        Tone: {insights.tone}
        Target Audience: {insights.target_audience}
        Value Proposition: {insights.value_proposition}
        Key Products: {', '.join(insights.key_products)}
        Brand Personality: {insights.brand_personality}
    
        For each meme concept, provide:
        1. The main message/joke
        2. The intended emotional response
        3. How it relates to the target audience
    
        Format the response as JSON array with structure:
        [{{"message": "string", "emotion": "string", "audience_relevance": "string"}}]"""
    
        response = llm.invoke([HumanMessage(content=prompt)])
    
        json_match = re.search(r'\[\s*{.*}\s*\]', response.content, re.DOTALL)
        if json_match:
            concepts_json = json_match.group(0)
            try:
                concepts = json.loads(concepts_json)
                state["meme_concepts"] = concepts
                state["selected_concepts"] = concepts[:3]  # Select top 3 concepts
                return state
            except json.JSONDecodeError:
                logger.debug("Failed to parse JSON response")
    
        return state
    
    async def get_meme_templates() -> Dict[str, TemplateInfo]:
        """
        Fetches available meme templates from Memegen.link API.
    
        Returns:
            Dict[str, TemplateInfo]: Dictionary of template information keyed by template ID
    
        Notes:
            - Fetches templates from Memegen.link API
            - Filters templates to those with 2 or fewer lines
            - Randomly selects 20 templates
            - Converts API response to TemplateInfo objects
            - Includes template metadata like name, description, and example text
        """
    
        async with aiohttp.ClientSession() as session:
                async with session.get("https://api.memegen.link/templates/") as response:
                    all_templates = await response.json()
            
                    filtered_templates = [
                        template for template in all_templates
                        if template.get("lines", 0) <= 2
                    ]
            
                    selected_templates = random.sample(filtered_templates, min(20, len(filtered_templates)))
            
                    template_dict = {
                        template["id"]: TemplateInfo(
                            template_id=template["id"],
                            name=template["name"],
                            blank_template_api_link=template["blank"],
                            description=f"{template['name']} meme with {template['lines']} text lines.",
                            example_text_1=template.get('example', {}).get('text', [''])[0] or '',
                            example_text_2=template.get('example', {}).get('text', ['', ''])[1] if len(template.get('example', {}).get('text', [])) > 1 else '',
                            lines=template["lines"],
                            keywords=template.get("keywords", [])
                        )
                        for template in selected_templates
                    }
                    return template_dict
            
        logger.success(format_json(result))
    def select_meme_templates(state: GraphState) -> GraphState:
        """
        Selects appropriate meme templates for each concept.
    
        Args:
            state (GraphState): Current state containing selected_concepts and available_templates
    
        Returns:
            GraphState: Updated state with selected_memes added
    
        Notes:
            - Creates simplified template descriptions for LLM
            - Matches concepts with appropriate templates
            - Falls back to random selection if no match found
            - Handles template selection for each concept
            - Creates structured meme objects with template info
        """
    
        concepts = state["selected_concepts"]
        templates = state["available_templates"]
        selected_memes = {}
    
        template_descriptions = [
            {
                'template_id': template_id,
                'name': template_data.name,
                'description': template_data.description,
                'lines': template_data.lines
            }
            for template_id, template_data in templates.items()
        ]
    
        for idx, concept in enumerate(concepts):
            prompt = f"""Select a meme template that best fits this concept:
    
            Concept:
            - Message: {concept['message']}
            - Emotion: {concept['emotion']}
            - Audience Relevance: {concept['audience_relevance']}
    
            Available Templates:
            {json.dumps(template_descriptions, indent=2)}
    
            Return only the template ID that best matches the concept's message and emotion."""
    
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                template_id = response.content.strip().strip('"').strip("'").lower()
    
                if template_id not in templates:
                    template_id = random.choice(list(templates.keys()))
    
                selected_memes[f"meme_{idx+1}"] = {
                    "meme_id": f"meme_{idx+1}",
                    "template_id": template_id,
                    "concept": concept,
                    "template_info": templates[template_id],
                    "blank_template_api_link": templates[template_id].blank_template_api_link,
                    "is_text_element1_filled": True,
                    "is_text_element2_filled": templates[template_id].lines >= 2
                }
    
            except Exception as e:
                logger.debug(f"Error selecting template: {str(e)}")
                continue
    
        state["selected_memes"] = selected_memes
        return state
    
    def generate_text_elements(state: GraphState) -> GraphState:
        """
        Generates meme text based on selected concepts and templates.
    
        Args:
            state (GraphState): Current state containing selected_memes and company_context
    
        Returns:
            GraphState: Updated state with pre_generated_memes added
    
        Notes:
            - Generates appropriate text for each template
            - Considers template format and number of lines
            - Maintains brand tone and target audience
            - Creates concise, punchy text elements
            - Handles errors gracefully for each meme
        """
    
        selected_memes = state["selected_memes"]
        context = state["company_context"]
        pre_generated_memes = {}
    
        for meme_id, meme in selected_memes.items():
            concept = meme["concept"]
            template_info = meme["template_info"]
    
            prompt = f"""Create text for a meme based on this template and concept:
    
            Template: {template_info.name}
            Number of lines: {template_info.lines}
            Example Text 1: {template_info.example_text_1}
            Example Text 2: {template_info.example_text_2}
            Concept Message: {concept['message']}
            Emotion: {concept['emotion']}
    
            Company Context:
            Target Audience: {context.target_audience}
            Brand Tone: {context.tone}
    
            Return ONLY the text lines, one per line. Keep each line concise and punchy.
    
            """
    
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                text_elements = response.content.strip().split('\n')
    
                generated_text1 = text_elements[0] if len(text_elements) > 0 else ""
                generated_text2 = text_elements[1] if len(text_elements) > 1 else ""
    
                pre_generated_memes[meme_id] = {
                    **meme,
                    "generated_text_element1": generated_text1,
                    "generated_text_element2": generated_text2
                }
    
            except Exception as e:
                logger.debug(f"Error generating text: {str(e)}")
                continue
    
        state["pre_generated_memes"] = pre_generated_memes
        return state
    
    def create_meme_url(state: GraphState) -> GraphState:
        """
        Creates final meme URLs using the Memegen.link API format.
    
        Args:
            state (GraphState): Current state containing pre_generated_memes
    
        Returns:
            GraphState: Updated state with generated_memes added
    
        Notes:
            - Processes text elements for URL compatibility
            - Handles URL encoding of text
            - Extracts and manages file extensions
            - Constructs final meme URLs
            - Maintains all meme metadata in state
        """
    
        pre_generated_memes = state["pre_generated_memes"]
        generated_memes = {}
    
        for meme_id, meme in pre_generated_memes.items():
            text1 = quote(meme["generated_text_element1"].replace(' ', '_'))
            text2 = quote(meme["generated_text_element2"].replace(' ', '_'))
    
            template_info = meme["template_info"]
            base_url = template_info.blank_template_api_link
    
            extension = os.path.splitext(base_url)[1]
            base_url = base_url.rsplit('.', 1)[0]
    
            final_url = f"{base_url}/{text1}/{text2}{extension}"
    
            generated_memes[meme_id] = {
                **meme,
                "final_url": final_url,
                "text_elements": [text1, text2]
            }
    
        state["generated_memes"] = generated_memes
        return state
    
    async def display_meme(url: str):
        """
        Displays a meme from a given URL.
    
        Args:
            url (str): URL of the meme to display
    
        Returns:
            Optional[Image.Image]: PIL Image object if successful, None if failed
    
        Notes:
            - Fetches image data asynchronously
            - Converts bytes to PIL Image
            - Handles HTTP errors
            - Reports failures without crashing
        """
    
        try:
            async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            image = Image.open(BytesIO(image_data))
                            return image
                        else:
                            logger.debug(f"Failed to fetch image: Status {response.status}")
                            return None
            logger.success(format_json(result))
        except Exception as e:
            logger.debug(f"Error displaying meme: {str(e)}")
            return None
    
    """
    ## Set Up LangGraph Workflow
    
    Define the LangGraph workflow by adding nodes and edges.
    """
    logger.info("## Set Up LangGraph Workflow")
    
    workflow = Graph()
    
    workflow.add_node("get_website_content", get_website_content)
    workflow.add_node("analyze_company", analyze_company_insights)
    workflow.add_node("generate_concepts", generate_meme_concepts)
    workflow.add_node("select_templates", select_meme_templates)
    workflow.add_node("generate_text", generate_text_elements)
    workflow.add_node("create_url", create_meme_url)
    
    workflow.add_edge("get_website_content", "analyze_company")
    workflow.add_edge("analyze_company", "generate_concepts")
    workflow.add_edge("generate_concepts", "select_templates")
    workflow.add_edge("select_templates", "generate_text")
    workflow.add_edge("generate_text", "create_url")
    workflow.add_edge("create_url", END)
    
    workflow.set_entry_point("get_website_content")
    
    app = workflow.compile()
    
    """
    ## Display Graph Structure
    """
    logger.info("## Display Graph Structure")
    
    display(
        IPImage(
            app.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )
    
    """
    ## Run Workflow Function
    
    Define a function to run the workflow and display results.
    """
    logger.info("## Run Workflow Function")
    
    async def run_workflow(website_url: str):
        """
        Runs the complete meme generation workflow.
    
        Args:
            website_url (str): URL of the company website to analyze
    
        Returns:
            Complete workflow results or None if failed
    
        Notes:
            - Initializes workflow with available templates
            - Sets up initial state
            - Runs complete LangGraph workflow
            - Displays results including:
                - Company analysis
                - Generated memes
                - Preview images
            - Handles and reports errors
            - Returns full result data
        """
    
        logger.debug("Loading meme templates...")
        available_templates = await get_meme_templates()
        logger.success(format_json(available_templates))
    
        initial_state = {
            "messages": [],
            "website_url": website_url,
            "website_content": "",
            "company_context": {
                "tone": "",
                "target_audience": "",
                "value_proposition": "",
                "key_products": [],
                "brand_personality": ""
            },
            "meme_concepts": [],
            "selected_concepts": [],
            "selected_memes": {},
            "generated_memes": {},
            "available_templates": available_templates
        }
    
        try:
    
            result = await app.ainvoke(initial_state)
            logger.success(format_json(result))
    
            if isinstance(result, dict) and "company_context" in result:
                logger.debug("\nCompany Analysis:")
                logger.debug("")
                for key, value in result["company_context"]:
                    logger.debug(f"{key.title()}: {value}")
    
                if "generated_memes" in result:
                    logger.debug("\nGenerated Memes:")
                    for meme_id, meme_info in result["generated_memes"].items():
                        logger.debug(f"\n{meme_id.upper()}:")
                        logger.debug("")
                        template_info = meme_info.get('template_info', {})
                        logger.debug(f"Template: {template_info.name}")
                        logger.debug(f"Blank template image: {template_info.blank_template_api_link}")
                        logger.debug("")
                        logger.debug(f"Concept message: {meme_info['concept']['message']}")
                        logger.debug(f"Concept emotion: {meme_info['concept']['emotion']}")
                        logger.debug(f"Concept audience relevance: {meme_info['concept']['audience_relevance']}")
                        logger.debug("")
                        logger.debug(f"Generated captions:")
                        logger.debug(meme_info["generated_text_element1"])
                        logger.debug(meme_info["generated_text_element2"])
                        logger.debug(f"URL: {meme_info['final_url']}")
                        logger.debug("")
                        meme_image = await display_meme(meme_info["final_url"])
                        logger.success(format_json(meme_image))
                        if meme_image:
                          display(meme_image)
                        else:
                          logger.debug(f"Failed to display {meme_id}")
                        logger.debug("--------------------------------------------------------------------------")
            return result
    
        except Exception as e:
            logger.debug(f"An error occurred: {str(e)}")
            traceback.print_exc()
            return None
    
    """
    ## Execute Workflow
    
    Run the workflow with a sample query.
    """
    logger.info("## Execute Workflow")
    
    website_url = "https://www.langchain.com/"
    logger.debug(f"Generating memes for: {website_url}")
    result = await run_workflow(website_url)
    logger.success(format_json(result))
    
    if result:
        logger.debug("\nWorkflow completed successfully!")
    else:
        logger.debug("\nWorkflow failed. Please check the error messages above.")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())