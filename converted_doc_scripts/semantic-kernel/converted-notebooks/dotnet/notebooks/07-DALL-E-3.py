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
# Generating images with AI

This notebook demonstrates how to use Ollama DALL-E 3 to generate images, in combination with other LLM features like text and embedding generation.

Here, we use Chat Completion to generate a random image description and DALL-E 3 to create an image from that description, showing the image inline.

Lastly, the notebook asks the user to describe the image. The embedding of the user's description is compared to the original description, using Cosine Similarity, and returning a score from 0 to 1, where 1 means exact match.
"""
logger.info("# Generating images with AI")

// Usual setup: importing Semantic Kernel SDK and SkiaSharp, used to display images inline.



using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.TextToImage;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Connectors.Ollama;
using System.Numerics.Tensors;

"""
# Setup, using three AI services: images, text, embedding

The notebook uses:

* **Ollama Dall-E 3** to transform the image description into an image
* **text-embedding-ada-002** to compare your guess against the real image description

**Note:**: For Azure Ollama, your endpoint should have DALL-E API enabled.
"""
logger.info("# Setup, using three AI services: images, text, embedding")

using Kernel = Microsoft.SemanticKernel.Kernel;


// Load Ollama credentials from config/settings.json
var (useAzureOllama, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

// Configure the three AI features: text embedding (using Ada), chat completion, image generation (DALL-E 3)
var builder = Kernel.CreateBuilder();

if(useAzureOllama)
{
    builder.AddAzureOllamaTextEmbeddingGeneration("text-embedding-ada-002", azureEndpoint, apiKey);
    builder.AddAzureOllamaChatCompletion(model, azureEndpoint, apiKey);
    builder.AddAzureOllamaTextToImage("dall-e-3", azureEndpoint, apiKey);
}
else
{
    builder.AddOllamaTextEmbeddingGeneration("text-embedding-ada-002", apiKey, orgId);
    builder.AddOllamaChatCompletion(model, apiKey, orgId);
    builder.AddOllamaTextToImage(apiKey, orgId);
}

var kernel = builder.Build();

// Get AI service instance used to generate images
var dallE = kernel.GetRequiredService<ITextToImageService>();

// Get AI service instance used to extract embedding from a text
var textEmbedding = kernel.GetRequiredService<ITextEmbeddingGenerationService>();

"""
# Generate a (random) image with DALL-E 3

**genImgDescription** is a Semantic Function used to generate a random image description. 
The function takes in input a random number to increase the diversity of its output.

The random image description is then given to **Dall-E 3** asking to create an image.
"""
logger.info("# Generate a (random) image with DALL-E 3")

var prompt = @"
Think about an artificial object correlated to number {{$input}}.
Describe the image with one detailed sentence. The description cannot contain numbers.";

var executionSettings = new OllamaPromptExecutionSettings
{
    MaxTokens = 256,
    Temperature = 1
};

// Create a semantic function that generate a random image description.
var genImgDescription = kernel.CreateFunctionFromPrompt(prompt, executionSettings);

var random = new Random().Next(0, 200);
async def run_async_code_1ad406ab():
    async def run_async_code_68df749f():
        var imageDescriptionResult = await kernel.InvokeAsync(genImgDescription, new() { ["input"] = random });
        return var imageDescriptionResult
    var imageDescriptionResult = asyncio.run(run_async_code_68df749f())
    logger.success(format_json(var imageDescriptionResult))
    return var imageDescriptionResult
var imageDescriptionResult = asyncio.run(run_async_code_1ad406ab())
logger.success(format_json(var imageDescriptionResult))
var imageDescription = imageDescriptionResult.ToString();

// Use DALL-E 3 to generate an image. Ollama in this case returns a URL (though you can ask to return a base64 image)
async def run_async_code_1a6b35b5():
    async def run_async_code_87b64a2d():
        var imageUrl = await dallE.GenerateImageAsync(imageDescription.Trim(), 1024, 1024);
        return var imageUrl
    var imageUrl = asyncio.run(run_async_code_87b64a2d())
    logger.success(format_json(var imageUrl))
    return var imageUrl
var imageUrl = asyncio.run(run_async_code_1a6b35b5())
logger.success(format_json(var imageUrl))

async def run_async_code_e96fcfbd():
    await SkiaUtils.ShowImage(imageUrl, 1024, 1024);
    return 
 = asyncio.run(run_async_code_e96fcfbd())
logger.success(format_json())

"""
# Let's play a guessing game

Try to guess what the image is about, describing the content.

You'll get a score at the end 😉
"""
logger.info("# Let's play a guessing game")

// Prompt the user to guess what the image is
async def run_async_code_31785baa():
    async def run_async_code_194dd299():
        var guess = await InteractiveKernel.GetInputAsync("Describe the image in your words");
        return var guess
    var guess = asyncio.run(run_async_code_194dd299())
    logger.success(format_json(var guess))
    return var guess
var guess = asyncio.run(run_async_code_31785baa())
logger.success(format_json(var guess))

// Compare user guess with real description and calculate score
async def run_async_code_ba1c506e():
    async def run_async_code_6a18dc3a():
        var origEmbedding = await textEmbedding.GenerateEmbeddingsAsync(new List<string> { imageDescription } );
        return var origEmbedding
    var origEmbedding = asyncio.run(run_async_code_6a18dc3a())
    logger.success(format_json(var origEmbedding))
    return var origEmbedding
var origEmbedding = asyncio.run(run_async_code_ba1c506e())
logger.success(format_json(var origEmbedding))
async def run_async_code_7339aa07():
    async def run_async_code_0c93ad2c():
        var guessEmbedding = await textEmbedding.GenerateEmbeddingsAsync(new List<string> { guess } );
        return var guessEmbedding
    var guessEmbedding = asyncio.run(run_async_code_0c93ad2c())
    logger.success(format_json(var guessEmbedding))
    return var guessEmbedding
var guessEmbedding = asyncio.run(run_async_code_7339aa07())
logger.success(format_json(var guessEmbedding))
var similarity = TensorPrimitives.CosineSimilarity(origEmbedding.First().Span, guessEmbedding.First().Span);

Console.WriteLine($"Your description:\n{Utils.WordWrap(guess, 90)}\n");
Console.WriteLine($"Real description:\n{Utils.WordWrap(imageDescription.Trim(), 90)}\n");
Console.WriteLine($"Score: {similarity:0.00}\n\n");

//Uncomment this line to see the URL provided by Ollama
//Console.WriteLine(imageUrl);

logger.info("\n\n[DONE]", bright=True)