async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    import * as pl from "npm:nodejs-polars";
    import os
    import shutil
    import {
    import { ChatOllama } from "npm:@langchain/ollama";
    import { ChatPromptTemplate } from "npm:@langchain/core/prompts";
    import { MessagesPlaceholder } from "npm:@langchain/core/prompts";
    import { MongoClient } from "npm:mongodb";
    import { MongoDBAtlasVectorSearch } from "npm:@langchain/mongodb";
    import { MongoDBChatMessageHistory } from "npm:@langchain/mongodb";
    import { OllamaEmbeddings } from "npm:@langchain/ollama";
    import { RunnableWithMessageHistory } from "npm:@langchain/core/runnables";
    import { StringOutputParser } from "npm:@langchain/core/output_parsers";
    import { formatDocumentsAsString } from "npm:langchain/util/document";
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # Adding Chat History to your RAG Application using MongoDB and LangChain.js
    
    In this notebook, we will see how to use the new MongoDBChatMessageHistory in your RAG application.
    
    ## Step 1: Set up environment
    
    You may be used to Jupyter notebooks that use Python, but this notebook uses JavaScript, specifically [Deno](https://deno.land). 
    
    To run this notebook, you will need to install Deno and set up the [Deno Jupyter kernel](https://deno.com/blog/v1.37).
    
    Here are the basic instructions:
    
    - Install Deno: 
      - MacOS / Linux: `curl -fsSL https://deno.land/x/install/install.sh | sh`
      - Windows: `irm https://deno.land/install.ps1 | iex`
    - Install Python: Download and install from [python.org](https://www.python.org/downloads/)
    - Install Jupyter: `pip install jupyterlab`
    - Install Deno Jupyter kernel: `deno jupyter --unstable` or `deno jupyter --unstable --install` to force installation of the kernel.
    
    ## Step 2: Setup pre-requisites
    
    * Set the MongoDB connection string. Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.
    
    * Set the Ollama API key. Steps to obtain an API key as [here](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key)
    """
    logger.info("# Adding Chat History to your RAG Application using MongoDB and LangChain.js")
    
    const MONGODB_URI = "Enter your MongoDB connection string here";
    # const OPENAI_API_KEY = "Enter your Ollama API key here";
    
    """
    ## Step 3: Download the dataset
    
    We will be using MongoDB's [embedded_movies](https://huggingface.co/datasets/MongoDB/embedded_movies) dataset
    """
    logger.info("## Step 3: Download the dataset")
    
    const response = await fetch(
          "https://huggingface.co/datasets/MongoDB/embedded_movies/resolve/main/sample_mflix.embedded_movies.json?download=true"
        );
    logger.success(format_json(const response))
    let dataSet = await response.json();
    logger.success(format_json(let dataSet))
    
    
    const df = pl.readJSON(JSON.stringify(dataSet));
    // Previewing the contents of the data
    df.head(1);
    
    // Only keep records where the fullplot field is not null
    const filtered = df.filter(pl.pl.col("fullplot") !== null);
    // Renaming the embedding field to "embedding" -- required by LangChain
    const renamed = filtered.withColumnRenamed("plot_embedding", "embedding");
    
    """
    ## Step 4: Create a simple RAG chain using MongoDB as the vector store
    """
    logger.info("## Step 4: Create a simple RAG chain using MongoDB as the vector store")
    
    
    // Initialize MongoDB client
    const client = new MongoClient(MONGODB_URI, {appName="devrel.showcase.langchain_js_memory"});
    const DB_NAME = "langchain_chatbot";
    const COLLECTION_NAME = "data";
    const INDEX_NAME = "vector_index";
    const collection = client.db(DB_NAME).collection(COLLECTION_NAME);
    
    // Delete any existing records in the collection
    await collection.deleteMany({});
    
    // Data Ingestion
    const records = renamed.toRecords();
    const {insertedIds: _, ...result} = await collection.insertMany(records);
    logger.success(format_json(const {insertedIds: _, ...result}))
    console.log(result);
    
    // Using the text-embedding-ada-002 since that's what was used to create embeddings in the movies dataset
    const embeddings = new OllamaEmbeddings({
    #   openAIApiKey: OPENAI_API_KEY,
      modelName: "text-embedding-ada-002",
    });
    
    
    const dbConfig = {
      collection: collection,
      indexName: INDEX_NAME,
      textKey: "fullplot",
      embeddingKey: "embedding",
     };
    
    // Vector Store Creation
    const vectorStore = new MongoDBAtlasVectorSearch(embeddings, dbConfig);
    
    // Using the MongoDB vector store as a retriever in a RAG chain
    const retriever = vectorStore.asRetriever({
      searchType: "similarity",
      k: 5
    });
    
      RunnableSequence,
      RunnablePassthrough,
    } from "npm:@langchain/core/runnables";
    // Generate context using the retriever, and pass the user question through
    const system = `Answer the question based only on the following context:
      <context>
        {context}
      </context>
      If the answer is not in this context, please respond with "I don't know. Here is the context I was given: {context}"
    `;
    // Defining the chat prompt
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", system],
      ["human", "{question}"],
    ]);
    // Defining the model to be used for chat completion
    const model = new ChatOllama({
    #   openAIApiKey: OPENAI_API_KEY,
      temperature: 0,
    });
    // Naive RAG chain
    const naiveRagChain = RunnableSequence.from([
      {
        context: retriever.pipe(formatDocumentsAsString),
        question: new RunnablePassthrough(),
      },
      prompt,
      model,
      new StringOutputParser(),
    ]);
    
    await naiveRagChain.invoke("What is the best movie to watch when sad?");
    
    """
    ## Step 5: Create a RAG chain with chat history
    """
    logger.info("## Step 5: Create a RAG chain with chat history")
    
    
    const collectionHistory = client.db(DB_NAME).collection("history");
    
    const chatHistory = new MongoDBChatMessageHistory({
      collection: collectionHistory,
      sessionId: "1", // Unique identifier for the chat session
    });
    
    // Clear the chat history as needed for testing
    await chatHistory.clear();
    
    
    // Given a follow-up question and history, create a standalone question
    const standaloneSystemPrompt = `
    Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question.
    Do NOT answer the question, just reformulate it if needed, otherwise return it as is.
    Only return the final standalone question.
    `;
    
    const standaloneQuestionPrompt = ChatPromptTemplate.fromMessages([
      ["system", standaloneSystemPrompt],
      new MessagesPlaceholder("history"),
      ["human", "{question}"],
    ]);
    
    const questionChain = RunnableSequence.from([
      standaloneQuestionPrompt,
      model,
      new StringOutputParser(),
    ]);
    
    const retrieverChain = RunnablePassthrough.assign({
      context: questionChain.pipe(retriever).pipe(formatDocumentsAsString),
    })
    
    // Create a prompt that includes the context, history and the follow-up question
    const ragSystemPrompt = `Answer the question based only on the following context:
    {context}`;
    
    const ragPrompt = ChatPromptTemplate.fromMessages([
      ["system", ragSystemPrompt],
      new MessagesPlaceholder("history"),
      ["human", "{question}"],
    ]);
    
    // RAG chain
    const ragChain = RunnableSequence.from([
      retrieverChain,
      ragPrompt,
      model,
      new StringOutputParser(),
    ]);
    
    
    // RAG chain with history
    const withMessageHistory = new RunnableWithMessageHistory({
      runnable: ragChain,
      getMessageHistory: () => chatHistory,
      inputMessagesKey: "question",
      historyMessagesKey: "history",
    });
    
    await withMessageHistory.invoke(
      { question: "What is the best movie to watch when sad?" },
      { configurable: { sessionId: "1" } }
    );
    
    await withMessageHistory.invoke(
      { question: "Hmmm..I don't want to watch that one. Can you suggest something else?" },
      { configurable: { sessionId: "1" } }
    );
    
    await withMessageHistory.invoke(
      { question: "How about something more light?" },
      { configurable: { sessionId: "1" } }
    );
    
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