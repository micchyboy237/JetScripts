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
# RAG with BingSearch 

This notebook explains how to integrate Bing Search with the Semantic Kernel  to get the latest information from the internet.

To use Bing Search you simply need a Bing Search API key. You can get the API key by creating a [Bing Search resource](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource) in Azure. 

To learn more read the following [documentation](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview).

Prepare a Semantic Kernel instance first, loading also the AI backend settings defined in the [Setup notebook](0-AI-settings.ipynb):
"""
logger.info("# RAG with BingSearch")



"""
Enter your Bing Search Key in BING_KEY using `InteractiveKernel` method as introduced in [`.NET Interactive`](https://github.com/dotnet/interactive/blob/main/docs/kernels-overview.md)
"""
logger.info("Enter your Bing Search Key in BING_KEY using `InteractiveKernel` method as introduced in [`.NET Interactive`](https://github.com/dotnet/interactive/blob/main/docs/kernels-overview.md)")

using InteractiveKernel = Microsoft.DotNet.Interactive.Kernel;

async def run_async_code_37869857():
    string BING_KEY = (await InteractiveKernel.GetPasswordAsync("Please enter your Bing Search Key")).GetClearTextPassword();
    return 
 = asyncio.run(run_async_code_37869857())
logger.success(format_json())

"""
## Basic search plugin

The sample below shows how to create a plugin named SearchPlugin from an instance of `BingTextSearch`. Using `CreateWithSearch` creates a new plugin with a single Search function that calls the underlying text search implementation. The SearchPlugin is added to the Kernel which makes it available to be called during prompt rendering. The prompt template includes a call to `{{SearchPlugin.Search $query}}` which will invoke the SearchPlugin to retrieve results related to the current query. The results are then inserted into the rendered prompt before it is sent to the AI model.
"""
logger.info("## Basic search plugin")

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Data;
using Microsoft.SemanticKernel.Plugins.Web.Bing;
using Kernel = Microsoft.SemanticKernel.Kernel;

// Create a kernel with Ollama chat completion
var builder = Kernel.CreateBuilder();

// Configure AI backend used by the kernel
var (useAzureOllama, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();
if (useAzureOllama)
    builder.AddAzureOllamaChatCompletion(model, azureEndpoint, apiKey);
else
    builder.AddOllamaChatCompletion(model, apiKey, orgId);
var kernel = builder.Build();

// Create a text search using Bing search
var textSearch = new BingTextSearch(apiKey: BING_KEY);

// Build a text search plugin with Bing search and add to the kernel
var searchPlugin = textSearch.CreateWithSearch("SearchPlugin");
kernel.Plugins.Add(searchPlugin);

// Invoke prompt and use text search plugin to provide grounding information
var query = "What is the Semantic Kernel?";
var prompt = "{{SearchPlugin.Search $query}}. {{$query}}";
KernelArguments arguments = new() { { "query", query } };
async def run_async_code_1cdf5467():
    Console.WriteLine(await kernel.InvokePromptAsync(prompt, arguments));
    return 
 = asyncio.run(run_async_code_1cdf5467())
logger.success(format_json())

"""
## Search plugin with citations

The sample below repeats the pattern described in the previous section with a few notable changes:

1. `CreateWithGetTextSearchResults` is used to create a `SearchPlugin` which calls the `GetTextSearchResults` method from the underlying text search implementation.
2. The prompt template uses Handlebars syntax. This allows the template to iterate over the search results and render the name, value and link for each result.
3. The prompt includes an instruction to include citations, so the AI model will do the work of adding citations to the response.
"""
logger.info("## Search plugin with citations")

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Data;
using Microsoft.SemanticKernel.Plugins.Web.Bing;
using Microsoft.SemanticKernel.PromptTemplates.Handlebars;
using Kernel = Microsoft.SemanticKernel.Kernel;

// Create a kernel with Ollama chat completion
var builder = Kernel.CreateBuilder();

// Configure AI backend used by the kernel
var (useAzureOllama, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();
if (useAzureOllama)
    builder.AddAzureOllamaChatCompletion(model, azureEndpoint, apiKey);
else
    builder.AddOllamaChatCompletion(model, apiKey, orgId);
var kernel = builder.Build();

// Create a text search using Bing search
var textSearch = new BingTextSearch(apiKey: BING_KEY);

// Build a text search plugin with Bing search and add to the kernel
var searchPlugin = textSearch.CreateWithGetTextSearchResults("SearchPlugin");
kernel.Plugins.Add(searchPlugin);

// Invoke prompt and use text search plugin to provide grounding information
var query = "What is the Semantic Kernel?";
string promptTemplate = """
{{#with (SearchPlugin-GetTextSearchResults query)}}
    {{#each this}}
    Name: {{Name}}
    Value: {{Value}}
    Link: {{Link}}
    -----------------
    {{/each}}
{{/with}}

{{query}}

Include citations to the relevant information where it is referenced in the response.
""";
KernelArguments arguments = new() { { "query", query } };
HandlebarsPromptTemplateFactory promptTemplateFactory = new();
Console.WriteLine(await kernel.InvokePromptAsync(
    promptTemplate,
    arguments,
    templateFormat: HandlebarsPromptTemplateFactory.HandlebarsTemplateFormat,
    promptTemplateFactory: promptTemplateFactory
));

"""
## Search plugin with a filter

The samples shown so far will use the top ranked web search results to provide the grounding data. To provide more reliability in the data the web search can be restricted to only return results from a specified site.

The sample below builds on the previous one to add filtering of the search results.
A `TextSearchFilter` with an equality clause is used to specify that only results from the Microsoft Developer Blogs site (`site == 'devblogs.microsoft.com'`) are to be included in the search results.

The sample uses `KernelPluginFactory.CreateFromFunctions` to create the `SearchPlugin`.
A custom description is provided for the plugin.
The `ITextSearch.CreateGetTextSearchResults` extension method is used to create the `KernelFunction` which invokes the text search service.
"""
logger.info("## Search plugin with a filter")

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Data;
using Microsoft.SemanticKernel.PromptTemplates.Handlebars;
using Microsoft.SemanticKernel.Plugins.Web.Bing;

// Create a kernel with Ollama chat completion
var builder = Kernel.CreateBuilder();

// Configure AI backend used by the kernel
var (useAzureOllama, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();
if (useAzureOllama)
    builder.AddAzureOllamaChatCompletion(model, azureEndpoint, apiKey);
else
    builder.AddOllamaChatCompletion(model, apiKey, orgId);
var kernel = builder.Build();

// Create a text search using Bing search
var textSearch = new BingTextSearch(apiKey: BING_KEY);

// Create a filter to search only the Microsoft Developer Blogs site
var filter = new TextSearchFilter().Equality("site", "devblogs.microsoft.com");
var searchOptions = new TextSearchOptions() { Filter = filter };

// Build a text search plugin with Bing search and add to the kernel
var searchPlugin = KernelPluginFactory.CreateFromFunctions(
    "SearchPlugin", "Search Microsoft Developer Blogs site only",
    [textSearch.CreateGetTextSearchResults(searchOptions: searchOptions)]);
kernel.Plugins.Add(searchPlugin);

// Invoke prompt and use text search plugin to provide grounding information
var query = "What is the Semantic Kernel?";
string promptTemplate = """
{{#with (SearchPlugin-GetTextSearchResults query)}}
    {{#each this}}
    Name: {{Name}}
    Value: {{Value}}
    Link: {{Link}}
    -----------------
    {{/each}}
{{/with}}

{{query}}

Include citations to the relevant information where it is referenced in the response.
""";
KernelArguments arguments = new() { { "query", query } };
HandlebarsPromptTemplateFactory promptTemplateFactory = new();
Console.WriteLine(await kernel.InvokePromptAsync(
    promptTemplate,
    arguments,
    templateFormat: HandlebarsPromptTemplateFactory.HandlebarsTemplateFormat,
    promptTemplateFactory: promptTemplateFactory
));

logger.info("\n\n[DONE]", bright=True)