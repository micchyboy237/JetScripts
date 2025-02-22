from jet.libs.txtai import Agent, Embeddings
from jet.libs.txtai.pipeline import Textractor
from jet.libs.txtai.workflow import Workflow, Task
from IPython.display import display, Markdown


def install_dependencies():
    # egg=txtai[graph] autoawq
    !pip install git+https: // github.com/neuml/txtai


def create_agent_with_wikipedia():
    return Agent(
        tools=[{
            "name": "wikipedia",
            "description": "Searches a Wikipedia database",
            "provider": "huggingface-hub",
            "container": "neuml/txtai-wikipedia"
        }],
        llm="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        max_iterations=10,
    )


def generate_report_with_wikipedia(agent):
    answer = agent("""
    I'm bored 🥱. Think about 2-3 disparate topics and use those to search wikipedia to generate something fascinating.
    Write a report summarizing each article. Include a section with a list of article hyperlinks.
    Write the text as Markdown.
    """, maxlength=16000)
    display(Markdown(answer))


def create_agent_with_hfposts():
    return Agent(
        tools=[{
            "name": "hfposts",
            "description": "Searches a database of technical posts on Hugging Face",
            "provider": "huggingface-hub",
            "container": "neuml/txtai-hfposts"
        }],
        llm="Qwen/Qwen2.5-7B-Instruct-AWQ",
        max_iterations=10,
    )


def generate_report_with_hfposts(agent):
    answer = agent("""
    Read posts about medicine and write a report on what you learned.
    The report should be a Markdown table with the following columns.
     - Name
     - Description
     - Link to content
    Only include rows that have a valid web url.
    """, maxlength=16000)
    display(Markdown(answer))


def create_embeddings_agent():
    embeddings = Embeddings(
        path="intfloat/e5-large",
        instructions={"query": "query: ", "data": "passage: "},
        content=True
    )
    textractor = Textractor(sections=True, headers={
                            "user-agent": "Mozilla/5.0"})

    def insert(elements):
        def upsert(elements):
            embeddings.upsert(elements)
            return elements
        workflow = Workflow([Task(textractor), Task(upsert)])
        list(workflow(elements))
        return f"{elements} inserted successfully"

    return Agent(
        tools=[insert, embeddings.search, "websearch"],
        llm="Qwen/Qwen2.5-7B-Instruct-AWQ",
        max_iterations=10
    )


def autonomous_embeddings_workflow(agent, topic):
    prompt = f"""
    Run the following process:
      1. Search your internal knowledge for {topic}
      2. If not found, find relevant urls and insert those as a list of strings ONLY. Then rerun the search for {topic}.
      3. Write a detailed report about {topic} with Markdown sections covering the major topics. Include a section with hyperlink references.
    """
    answer = agent(prompt)
    display(Markdown(answer))


def main():
    install_dependencies()

    agent_wikipedia = create_agent_with_wikipedia()
    generate_report_with_wikipedia(agent_wikipedia)

    agent_hfposts = create_agent_with_hfposts()
    generate_report_with_hfposts(agent_hfposts)

    agent_embeddings = create_embeddings_agent()
    autonomous_embeddings_workflow(agent_embeddings, "txtai")
    autonomous_embeddings_workflow(agent_embeddings, "openscholar")


if __name__ == "__main__":
    main()
