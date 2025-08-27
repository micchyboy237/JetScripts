import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Vector Stores and Embeddings

So far, we've mostly been treating the kernel as a stateless orchestration engine.
We send text into a model API and receive text out. 

In a [previous notebook](04-kernel-arguments-chat.ipynb), we used `kernel arguments` to pass in additional
text into prompts to enrich them with more data. This allowed us to create a basic chat experience. 

However, if you solely relied on kernel arguments, you would quickly realize that eventually your prompt
would grow so large that you would run into the model's token limit. What we need is a way to persist state
and build both short-term and long-term memory to empower even more intelligent applications. 

To do this, we dive into the key concept of `Vector Stores` in the Semantic Kernel.

More information can be found [here](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors).
"""
logger.info("# Vector Stores and Embeddings")

using Microsoft.SemanticKernel;
using Kernel = Microsoft.SemanticKernel.Kernel;


var builder = Kernel.CreateBuilder();

// Configure AI service credentials used by the kernel
var (useAzureOllama, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

if (useAzureOllama)
{
    builder.AddAzureOllamaTextEmbeddingGeneration("text-embedding-ada-002", azureEndpoint, apiKey);
}
else
{
    builder.AddOllamaTextEmbeddingGeneration("text-embedding-ada-002", apiKey, orgId);
}

var kernel = builder.Build();

"""
Package `Microsoft.Extensions.VectorData.Abstractions`, which we downloaded in a previous code snippet, contains all necessary abstractions to work with vector stores. 

Together with abstractions, we also need to use an implementation of a concrete database connector, such as Azure AI Search, Azure CosmosDB, Qdrant, Redis and so on. A list of supported connectors can be found [here](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/out-of-the-box-connectors/).

In this example, we are going to use the in-memory connector for demonstration purposes - `Microsoft.SemanticKernel.Connectors.InMemory`.

## Define your model

It all starts from defining your data model. In abstractions, there are three main data model property types:

1. Key
2. Data
3. Vector

In most cases, a data model contains one key property, multiple data and vector properties, but some connectors may have restrictions, for example when only one vector property is supported. 

Also, each connector supports a different set of property types. For more information about supported property types in each connector, visit the connector's page, which can be found [here](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/out-of-the-box-connectors/).

There are two ways how to define your data model - using attributes (declarative way) or record definition (imperative way).

Here is how a data model could look like with attributes:
"""
logger.info("## Define your model")

using Microsoft.Extensions.VectorData;

public sealed class Glossary
{
    [VectorStoreRecordKey]
    public ulong Key { get; set; }

    [VectorStoreRecordData]
    public string Term { get; set; }

    [VectorStoreRecordData]
    public string Definition { get; set; }

    [VectorStoreRecordVector(Dimensions: 1536)]
    public ReadOnlyMemory<float> DefinitionEmbedding { get; set; }
}

"""
More information about each attribute and its properties can be found [here](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/defining-your-data-model#attributes).

There could be a case when you can't modify the existing class with attributes. In this case, you can define a separate record definition with all the information about your properties. Note that the defined data model class is still required in this case:
"""
logger.info("More information about each attribute and its properties can be found [here](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/defining-your-data-model#attributes).")

public sealed class GlossaryWithoutAttributes
{
    public ulong Key { get; set; }

    public string Term { get; set; }

    public string Definition { get; set; }

    public ReadOnlyMemory<float> DefinitionEmbedding { get; set; }
}

var recordDefinition = new VectorStoreRecordDefinition()
{
    Properties = new List<VectorStoreRecordProperty>()
    {
        new VectorStoreRecordKeyProperty("Key", typeof(ulong)),
        new VectorStoreRecordDataProperty("Term", typeof(string)),
        new VectorStoreRecordDataProperty("Definition", typeof(string)),
        new VectorStoreRecordVectorProperty("DefinitionEmbedding", typeof(ReadOnlyMemory<float>)) { Dimensions = 1536 }
    }
};

"""
## Define main components

As soon as you define your data model with either attributes or the record definition approach, you can start using it with your database of choice. 

There are a couple of abstractions that allow you to work with your database and collections:

1. `IVectorStoreRecordCollection<TKey, TRecord>` - represents a collection. This collection may or may not exist, and the interface provides methods to check if the collection exists, create it or delete it. The interface also provides methods to upsert, get and delete records. Finally, the interface inherits from `IVectorizedSearch<TRecord>` providing vector search capabilities.
2. `IVectorStore` - contains operations that spans across all collections in the vector store, e.g. `ListCollectionNames`. It also provides the ability to get `IVectorStoreRecordCollection<TKey, TRecord>` instances.

Each connector has extension methods to register your vector store and collection using DI - `services.AddInMemoryVectorStore()` or `services.AddInMemoryVectorStoreRecordCollection("collection-name")`. 

It's also possible to initialize these instances directly, which we are going to do in this notebook for simplicity:
"""
logger.info("## Define main components")

using Microsoft.SemanticKernel.Connectors.InMemory;


// Define vector store
var vectorStore = new InMemoryVectorStore();

// Get a collection instance using vector store
var collection = vectorStore.GetCollection<ulong, Glossary>("skglossary");

// Get a collection instance by initializing it directly
var collection2 = new InMemoryVectorStoreRecordCollection<ulong, Glossary>("skglossary");

"""
Initializing a collection instance will allow you to work with your collection and data, but it doesn't mean that this collection already exists in a database. To ensure you are working with existing collection, you can create it if it doesn't exist:
"""
logger.info("Initializing a collection instance will allow you to work with your collection and data, but it doesn't mean that this collection already exists in a database. To ensure you are working with existing collection, you can create it if it doesn't exist:")

async def run_async_code_ac8c3526():
    await collection.CreateCollectionIfNotExistsAsync();
    return 
 = asyncio.run(run_async_code_ac8c3526())
logger.success(format_json())

"""
Now, since we just created a new collection, it is empty, so we want to insert some records using the data model we defined above:
"""
logger.info("Now, since we just created a new collection, it is empty, so we want to insert some records using the data model we defined above:")

var glossaryEntries = new List<Glossary>()
{
    new Glossary()
    {
        Key = 1,
        Term = "API",
        Definition = "Application Programming Interface. A set of rules and specifications that allow software components to communicate and exchange data."
    },
    new Glossary()
    {
        Key = 2,
        Term = "Connectors",
        Definition = "Connectors allow you to integrate with various services provide AI capabilities, including LLM, AudioToText, TextToAudio, Embedding generation, etc."
    },
    new Glossary()
    {
        Key = 3,
        Term = "RAG",
        Definition = "Retrieval Augmented Generation - a term that refers to the process of retrieving additional data to provide as context to an LLM to use when generating a response (completion) to a user's question (prompt)."
    }
};

"""
If we want to perform a vector search on our records in the database, initializing just the key and data properties is not enough, we also need to generate and initialize vector properties. For that, we can use `ITextEmbeddingGenerationService` which we already registered above.

The line `#pragma warning disable SKEXP0001` is required because `ITextEmbeddingGenerationService` interface is experimental and may change in the future.
"""
logger.info("If we want to perform a vector search on our records in the database, initializing just the key and data properties is not enough, we also need to generate and initialize vector properties. For that, we can use `ITextEmbeddingGenerationService` which we already registered above.")

using Microsoft.SemanticKernel.Embeddings;


var textEmbeddingGenerationService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();

var tasks = glossaryEntries.Select(entry => Task.Run(async () =>
{
    async def run_async_code_f877d338():
        entry.DefinitionEmbedding = await textEmbeddingGenerationService.GenerateEmbeddingAsync(entry.Definition);
        return entry.DefinitionEmbedding
    entry.DefinitionEmbedding = asyncio.run(run_async_code_f877d338())
    logger.success(format_json(entry.DefinitionEmbedding))
}));

async def run_async_code_2af622df():
    await Task.WhenAll(tasks);
    return 
 = asyncio.run(run_async_code_2af622df())
logger.success(format_json())

"""
## Upsert records

Now our glossary records are ready to be inserted into the database. For that, we can use `collection.UpsertAsync` or `collection.UpsertBatchAsync` methods. Note that this operation is idempotent - if a record with a specific key doesn't exist, it will be inserted. If it already exists, it will be updated. As a result, we should receive the keys of the upserted records:
"""
logger.info("## Upsert records")

async def run_async_code_f0157aa6():
    await foreach (var key in collection.UpsertBatchAsync(glossaryEntries))
    return 
 = asyncio.run(run_async_code_f0157aa6())
logger.success(format_json())
{
    Console.WriteLine(key);
}

"""
## Get records by key

In order to ensure our records were upserted correctly, we can get these records by a key with `collection.GetAsync` or `collection.GetBatchAsync` methods. 

Both methods accept `GetRecordOptions` class as a parameter, where you can specify if you want to include vector properties in your response or not. Taking into account that the vector dimension value can be high, if you don't need to work with vectors in your code, it's recommended to not fetch them from the database. That's why `GetRecordOptions.IncludeVectors` property is `false` by default. 

In this example, we want to include vectors in the result to ensure that our data was upserted correctly:
"""
logger.info("## Get records by key")

var options = new GetRecordOptions() { IncludeVectors = true };

async def run_async_code_460c7114():
    await foreach (var record in collection.GetBatchAsync(keys: [1, 2, 3], options))
    return 
 = asyncio.run(run_async_code_460c7114())
logger.success(format_json())
{
    Console.WriteLine($"Key: {record.Key}");
    Console.WriteLine($"Term: {record.Term}");
    Console.WriteLine($"Definition: {record.Definition}");
    Console.WriteLine($"Definition Embedding: {JsonSerializer.Serialize(record.DefinitionEmbedding)}");
}

"""
## Perform a search

Since we ensured that our records are already in the database, we can perform a vector search with `collection.VectorizedSearchAsync` method. 

This method accepts the `VectorSearchOptions` class as a parameter, which allows configuration of the vector search operation - specify the maximum number of records to return, the number of results to skip before returning results, a search filter to use before doing the vector search and so on. More information about it can be found [here](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/vector-search#vector-search-options).

To perform a vector search, we need a vector generated from our query string:
"""
logger.info("## Perform a search")

var searchString = "I want to learn more about Connectors";
async def run_async_code_3d41612b():
    async def run_async_code_bda08081():
        var searchVector = await textEmbeddingGenerationService.GenerateEmbeddingAsync(searchString);
        return var searchVector
    var searchVector = asyncio.run(run_async_code_bda08081())
    logger.success(format_json(var searchVector))
    return var searchVector
var searchVector = asyncio.run(run_async_code_3d41612b())
logger.success(format_json(var searchVector))

"""
As soon as we have our search vector, we can perform a search operation. The result of the `collection.VectorizedSearchAsync` method will be a collection of records from the database with their search scores:
"""
logger.info("As soon as we have our search vector, we can perform a search operation. The result of the `collection.VectorizedSearchAsync` method will be a collection of records from the database with their search scores:")

async def run_async_code_5c249350():
    async def run_async_code_5c802c54():
        var searchResult = await collection.VectorizedSearchAsync(searchVector);
        return var searchResult
    var searchResult = asyncio.run(run_async_code_5c802c54())
    logger.success(format_json(var searchResult))
    return var searchResult
var searchResult = asyncio.run(run_async_code_5c249350())
logger.success(format_json(var searchResult))

async def run_async_code_a915efd5():
    await foreach (var result in searchResult.Results)
    return 
 = asyncio.run(run_async_code_a915efd5())
logger.success(format_json())
{
    Console.WriteLine($"Search score: {result.Score}");
    Console.WriteLine($"Key: {result.Record.Key}");
    Console.WriteLine($"Term: {result.Record.Term}");
    Console.WriteLine($"Definition: {result.Record.Definition}");
    Console.WriteLine("=========");
}

"""
## Additional information

There are more concepts related to the vector stores that will allow you to extend the capabilities. Each of them is described in more detail on the Microsoft Learn portal:

1. [Generic data model](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/generic-data-model) - allows to store and search data without a concrete data model type, using the generic data model instead.
2. [Custom mapper](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/how-to/vector-store-custom-mapper) - define a custom mapper for a specific connector, when the default mapping logic is not enough to work with a database.
3. [Code samples](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/code-samples) - end-to-end RAG sample, supporting multiple vectors in the same record, vector search with paging, interoperability with Langchain and more.
"""
logger.info("## Additional information")

logger.info("\n\n[DONE]", bright=True)