from haystack import Document, Pipeline
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from haystack_integrations.components.embedders.cohere import CohereDocumentEmbedder
from haystack_integrations.components.embedders.cohere import CohereTextEmbedder
from haystack_integrations.components.generators.cohere import CohereGenerator
from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)


"""
# Should I Stay at This Hotel?

*Notebook by [Bilge Yucel](https://www.linkedin.com/in/bilge-yucel/)*

Multilingual Generative QA Using Cohere and Haystack

In this notebook, we'll delve into the details of multilingual retrieval and multilingual generation, and demonstrate how to build a **Retrieval Augmented Generation (RAG)** pipeline to generate answers from multilingual hotel reviews using [Cohere](https://cohere.com/) models and [Haystack](https://github.com/deepset-ai/haystack). 🏡

**Haystack Useful Sources**

* [Docs](https://docs.haystack.deepset.ai/docs/intro)
* [Tutorials](https://haystack.deepset.ai/tutorials)
* [Cookbooks](https://github.com/deepset-ai/haystack-cookbook)

> For Haystack 1.x version, check out [Article: Multilingual Generative Question Answering with Haystack and Cohere](https://haystack.deepset.ai/blog/multilingual-qa-with-cohere)

## Installation

Let's start by installing [Haystack's Cohere integration](https://haystack.deepset.ai/integrations/cohere):
"""
logger.info("# Should I Stay at This Hotel?")

# !pip install cohere-haystack

"""
## Storing Multilingual Embeddings

To create a question answering system for hotel reviews, the first thing we need is a document store. We’ll use an [`InMemoryDocumentStore`](https://docs.haystack.deepset.ai/v2.0/docs/inmemorydocumentstore) to save the hotel reviews along with their embeddings.
"""
logger.info("## Storing Multilingual Embeddings")


document_store = InMemoryDocumentStore()

"""
## Getting Cohere API Key

After signing up, you can [get a Cohere API key](https://dashboard.cohere.com/api-keys) for free to start using Cohere models.
"""
logger.info("## Getting Cohere API Key")

# from getpass import getpass

# COHERE_API_KEY = getpass("Enter Cohere API key:")

"""
## Creating an Indexing Pipeline

Let's create an indexing pipeline to write the hotel reviews from different languages to our document store. For this, we'll split the long reviews with [`DocumentSplitter`](https://docs.haystack.deepset.ai/v2.0/docs/documentsplitter) and create multilingual embeddings for each document using `embed-multilingual-v3.0` model with [`CohereDocumentEmbedder`](https://docs.haystack.deepset.ai/v2.0/docs/coheredocumentembedder).
"""
logger.info("## Creating an Indexing Pipeline")


documents = [Document(content="O ar condicionado de um dos quartos deu problema, mas levaram um ventilador para ser utilizado. Também por ser em uma área bem movimentada, o barulho da rua pode ser ouvido. Porém, eles deixam protetores auriculares para o uso. Também senti falta de um espelho de corpo inteiro no apartamento. Só havia o do banheiro que mostra apenas a parte superior do corpo."),
             Document(content="Durchgängig Lärm, weil direkt an der Partymeile; schmutziges Geschirr; unvollständige Küchenausstattung; Abzugshaube über Herd ging für zwei Stunden automatisch an und lies sich nicht abstellen; Reaktionen auf Anfragen entweder gar nicht oder unfreundlich"),
             Document(content="Das Personal ist sehr zuvorkommend! Über WhatsApp war man im guten Kontakt und konnte alles erfragen. Auch das Angebot des Shuttleservices war super und würde ich empfehlen - sehr unkompliziert! Unser Flug hatte Verspätung und der Shuttle hat auf uns gewartet. Die Lage zur Innenstadt ist sehr gut,jedoch ist die Fensterfront direkt zur Club-Straße deshalb war es nachts bis drei/vier Uhr immer recht laut. Die Kaffeemaschine oder auch die Couch hätten sauberer sein können. Ansonsten war das Appartement aber völlig ok."),
             Document(content="Super appartement. Juste au dessus de plusieurs bars qui ferment très tard. A savoir à l'avance. (Bouchons d'oreilles fournis !)"),
             Document(content="Zapach moczu przy wejściu do budynku, może warto zainstalować tam mocne światło na czujnik ruchu, dla gości to korzystne a dla kogoś kto chciałby zrobić tam coś innego niekorzystne :-). Świetne lokalizacje w centrum niestety są na to narażane."),
             Document(content="El apartamento estaba genial y muy céntrico, todo a mano. Al lado de la librería Lello y De la Torre de los clérigos. Está situado en una zona de marcha, así que si vais en fin de semana , habrá ruido, aunque a nosotros no nos molestaba para dormir"),
             Document(content="The keypad with a code is convenient and the location is convenient. Basically everything else, very noisy, wi-fi didn't work, check-in person didn't explain anything about facilities, shower head was broken, there's no cleaning and everything else one may need is charged."),
             Document(content="It is very central and appartement has a nice appearance (even though a lot IKEA stuff), *W A R N I N G** the appartement presents itself as a elegant and as a place to relax, very wrong place to relax - you cannot sleep in this appartement, even the beds are vibrating from the bass of the clubs in the same building - you get ear plugs from the hotel -> now I understand why -> I missed a trip as it was so loud and I could not hear the alarm next day due to the ear plugs.- there is a green light indicating 'emergency exit' just above the bed, which shines very bright at night - during the arrival process, you felt the urge of the agent to leave as soon as possible. - try to go to 'RVA clerigos appartements' -> same price, super quiet, beautiful, city center and very nice staff (not an agency)- you are basically sleeping next to the fridge, which makes a lot of noise, when the compressor is running -> had to switch it off - but then had no cool food and drinks. - the bed was somehow broken down - the wooden part behind the bed was almost falling appart and some hooks were broken before- when the neighbour room is cooking you hear the fan very loud. I initially thought that I somehow activated the kitchen fan"),
             Document(content="Un peu salé surtout le sol. Manque de service et de souplesse"),
             Document(content="De comfort zo centraal voor die prijs."),
             Document(content="Die Lage war sehr Zentral und man konnte alles sehenswertes zu Fuß erreichen. Wer am Wochenende nachts schlafen möchte, sollte diese Unterkunft auf keinen Fall nehmen. Party direkt vor der Tür so das man denkt, man schläft mitten drin. Sehr Sehr laut also und das bis früh 5 Uhr. Ab 7 kommt dann die Straßenreinigung die keineswegs leiser ist."),
             Document(content="Ótima escolha! Apartamento confortável e limpo! O RoofTop é otimo para beber um vinho! O apartamento é localizado entre duas ruas de movimento noturno. Porem as janelas, blindam 90% do barulho. Não nos incomodou"),
             Document(content="Nous avons passé un séjour formidable. Merci aux personnes , le bonjours à Ricardo notre taxi man, très sympathique. Je pense refaire un séjour parmi vous, après le confinement, tout était parfait, surtout leur gentillesse, aucune chaude négative. Je n'ai rien à redire de négative, Ils étaient a notre écoute, un gentil message tout les matins, pour nous demander si nous avions besoins de renseignement et savoir si tout allait bien pendant notre séjour."),
             Document(content="Boa localização. Bom pequeno almoço. A tv não se encontrava funcional."),
             Document(content="Céntrico. Muy cómodo para moverse y ver Oporto. Edificio con terraza propia en la última planta. Todo reformado y nuevo. Te traen un estupendo desayuno todas las mañanas al apartamento. Solo que se puede escuchar algo de ruido de la calle a primeras horas de la noche. Es un zona de ocio nocturno. Pero respetan los horarios.")
]

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=200))
indexing_pipeline.add_component("embedder", CohereDocumentEmbedder(api_key=Secret.from_token(COHERE_API_KEY), model="embed-multilingual-v3.0"))
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

indexing_pipeline.run({"splitter": {"documents": documents}})

"""
## Building a RAG Pipeline

Now that we have multilingual embeddings indexed in our document store, we'll create a pipeline where users interact the most: Retrieval-Augmented Generation (RAG) Pipeline.

A RAG pipeline consists of two parts: document retrieval and answer generation.

### Multilingual Document Retrieval

In the document retrieval step of a RAG pipeline, [`CohereTextEmbedder`](https://docs.haystack.deepset.ai/v2.0/docs/coheretextembedder) creates an embedding for the query in the multilingual vector space and [`InMemoryEmbeddingRetriever`](https://docs.haystack.deepset.ai/v2.0/docs/inmemoryembeddingretriever) retrieves the most similar *top_k* documents to the query from the document store. In our case, the retrieved documents will be hotel reviews.

### Multilingual Answer Generation
In the generation step of the RAG pipeline, we'll use `command` model of Cohere with [`CohereGenerator`](https://docs.haystack.deepset.ai/v2.0/docs/coheregenerator) to generate an answer based on the retrieved documents.

Let’s create a prompt template to use for hotel reviews. In this template, we’ll have two prompt variables: `{{documents}}` and `{{question}}`. These variables will later be filled with the user question and the retrieved hotel reviews outputted from the retriever.
"""
logger.info("## Building a RAG Pipeline")


template = """
You will be provided with reviews in multiple languages for an accommodation.
Create a concise and informative answer for a given question based solely on the given reviews.

\nReviews:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

\nQuestion: {{question}};
\nAnswer:
"""
rag_pipe = Pipeline()
rag_pipe.add_component("embedder", CohereTextEmbedder(api_key=Secret.from_token(COHERE_API_KEY), model="embed-multilingual-v3.0"))
rag_pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=3))
rag_pipe.add_component("prompt_builder", PromptBuilder(template=template))
rag_pipe.add_component("llm", CohereGenerator(api_key=Secret.from_token(COHERE_API_KEY), model="command"))
rag_pipe.connect("embedder.embedding", "retriever.query_embedding")
rag_pipe.connect("retriever", "prompt_builder.documents")
rag_pipe.connect("prompt_builder", "llm")

"""
## Asking a Question

Learn if this hotel is a suitable place this stay with your questions!
"""
logger.info("## Asking a Question")

question = "Is this place too noisy to sleep?"
result = rag_pipe.run({
    "embedder": {"text": question},
    "prompt_builder": {"question": question}
})

logger.debug(result["llm"]["replies"][0])

"""
### Other questions you can try 👇
"""
logger.info("### Other questions you can try 👇")

question = "What are the problems about this place?"
result = rag_pipe.run({
    "embedder": {"text": question},
    "prompt_builder": {"question": question}
})

logger.debug(result["llm"]["replies"][0])

question = "What is good about this place?"
result = rag_pipe.run({
    "embedder": {"text": question},
    "prompt_builder": {"question": question}
})

logger.debug(result["llm"]["replies"][0])

question = "Should I stay at this hotel?"
result = rag_pipe.run({
    "embedder": {"text": question},
    "prompt_builder": {"question": question}
})

logger.debug(result["llm"]["replies"][0])

question = "How is the wifi?"
result = rag_pipe.run({
    "embedder": {"text": question},
    "prompt_builder": {"question": question}
})

logger.debug(result["llm"]["replies"][0])

question = "Are there pubs near by?"
result = rag_pipe.run({
    "embedder": {"text": question},
    "prompt_builder": {"question": question}
})

logger.debug(result["llm"]["replies"][0])

logger.info("\n\n[DONE]", bright=True)