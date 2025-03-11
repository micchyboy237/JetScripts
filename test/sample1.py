from jet.llm.ollama.base_langchain import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain_chroma.vectorstores import Chroma
from jet.file.utils import load_file, save_file
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


if __name__ == "__main__":
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/langchain/cookbook/generated/RAPTOR"
    all_texts: list[str] = load_file(f"{base_dir}/all_texts.json")

    embed_model = "nomic-embed-text"
    llm_model = "llama3.1"

    embd = OllamaEmbeddings(model=embed_model)
    model = ChatOllama(model=llm_model)

    vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd)
    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    query = "How many seasons and episodes does ”I’ll Become a Villainess Who Goes Down in History” anime have?"
    result = rag_chain.invoke(query)
    logger.newline()
    logger.info("Result (rag_chain.invoke):")
    logger.debug(query)
    logger.success(result)

    query_result_path = "generated/RAPTOR/query_result.json"
    save_file({
        "query": query,
        "result": result
    }, query_result_path)
