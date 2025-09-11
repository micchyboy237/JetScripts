from jet.logger import logger
from langchain_community.embeddings import AscendEmbeddings
import os
import shutil

async def main():
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger.basicConfig(filename=log_file)
    logger.info(f"Logs: {log_file}")
    
    PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    
    model = AscendEmbeddings(
        model_path="/root/.cache/modelscope/hub/yangjhchs/acge_text_embedding",
        device_id=0,
        query_instruction="Represend this sentence for searching relevant passages: ",
    )
    emb = model.embed_query("hellow")
    logger.debug(emb)
    
    doc_embs = model.embed_documents(
        ["This is a content of the document", "This is another document"]
    )
    logger.debug(doc_embs)
    
    model.aembed_query("hellow")
    
    await model.aembed_query("hellow")
    
    model.aembed_documents(
        ["This is a content of the document", "This is another document"]
    )
    
    await model.aembed_documents(
        ["This is a content of the document", "This is another document"]
    )
    
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