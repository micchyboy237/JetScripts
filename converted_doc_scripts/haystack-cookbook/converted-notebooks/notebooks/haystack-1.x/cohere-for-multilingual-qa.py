from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import AnswerParser, PromptNode, PromptTemplate
from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.pipelines import Pipeline
from haystack.schema import Document
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


"""
# Should I Stay at This Hotel?

*Notebook by [Bilge Yucel](https://www.linkedin.com/in/bilge-yucel/)*

> Check out [Article: Multilingual Generative Question Answering with Haystack and Cohere](https://haystack.deepset.ai/blog/multilingual-qa-with-cohere) for the detailed explanation of this notebook

Multilingual Generative QA Using Cohere and Haystack

In this notebook, we'll delve into the details of multilingual retrieval and multilingual generation, and demonstrate how to build a **Retrieval Augmented Generation (RAG)** pipeline to generate answers from multilingual hotel reviews using [Cohere](https://cohere.com/) models and [Haystack](https://github.com/deepset-ai/haystack). 🏡
"""
logger.info("# Should I Stay at This Hotel?")

# !pip install farm-haystack[inference]

# from getpass import getpass

# COHERE_API_KEY = getpass("Enter Cohere API key:")



document_store = InMemoryDocumentStore(embedding_dim=768, similarity= "dot_product")
retriever = EmbeddingRetriever(
    embedding_model="embed-multilingual-v2.0",
    document_store=document_store,
    api_key=COHERE_API_KEY
)
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=200,
    split_respect_sentence_boundary=True,
)

documents = [Document("O ar condicionado de um dos quartos deu problema, mas levaram um ventilador para ser utilizado. Também por ser em uma área bem movimentada, o barulho da rua pode ser ouvido. Porém, eles deixam protetores auriculares para o uso. Também senti falta de um espelho de corpo inteiro no apartamento. Só havia o do banheiro que mostra apenas a parte superior do corpo."),
             Document("Durchgängig Lärm, weil direkt an der Partymeile; schmutziges Geschirr; unvollständige Küchenausstattung; Abzugshaube über Herd ging für zwei Stunden automatisch an und lies sich nicht abstellen; Reaktionen auf Anfragen entweder gar nicht oder unfreundlich"),
             Document("Das Personal ist sehr zuvorkommend! Über WhatsApp war man im guten Kontakt und konnte alles erfragen. Auch das Angebot des Shuttleservices war super und würde ich empfehlen - sehr unkompliziert! Unser Flug hatte Verspätung und der Shuttle hat auf uns gewartet. Die Lage zur Innenstadt ist sehr gut,jedoch ist die Fensterfront direkt zur Club-Straße deshalb war es nachts bis drei/vier Uhr immer recht laut. Die Kaffeemaschine oder auch die Couch hätten sauberer sein können. Ansonsten war das Appartement aber völlig ok."),
             Document("Super appartement. Juste au dessus de plusieurs bars qui ferment très tard. A savoir à l'avance. (Bouchons d'oreilles fournis !)"),
             Document("Zapach moczu przy wejściu do budynku, może warto zainstalować tam mocne światło na czujnik ruchu, dla gości to korzystne a dla kogoś kto chciałby zrobić tam coś innego niekorzystne :-). Świetne lokalizacje w centrum niestety są na to narażane."),
             Document("El apartamento estaba genial y muy céntrico, todo a mano. Al lado de la librería Lello y De la Torre de los clérigos. Está situado en una zona de marcha, así que si vais en fin de semana , habrá ruido, aunque a nosotros no nos molestaba para dormir"),
             Document("The keypad with a code is convenient and the location is convenient. Basically everything else, very noisy, wi-fi didn't work, check-in person didn't explain anything about facilities, shower head was broken, there's no cleaning and everything else one may need is charged."),
             Document("It is very central and appartement has a nice appearance (even though a lot IKEA stuff), *W A R N I N G** the appartement presents itself as a elegant and as a place to relax, very wrong place to relax - you cannot sleep in this appartement, even the beds are vibrating from the bass of the clubs in the same building - you get ear plugs from the hotel -> now I understand why -> I missed a trip as it was so loud and I could not hear the alarm next day due to the ear plugs.- there is a green light indicating 'emergency exit' just above the bed, which shines very bright at night - during the arrival process, you felt the urge of the agent to leave as soon as possible. - try to go to 'RVA clerigos appartements' -> same price, super quiet, beautiful, city center and very nice staff (not an agency)- you are basically sleeping next to the fridge, which makes a lot of noise, when the compressor is running -> had to switch it off - but then had no cool food and drinks. - the bed was somehow broken down - the wooden part behind the bed was almost falling appart and some hooks were broken before- when the neighbour room is cooking you hear the fan very loud. I initially thought that I somehow activated the kitchen fan"),
             Document("Un peu salé surtout le sol. Manque de service et de souplesse"),
             Document("De comfort zo centraal voor die prijs."),
             Document("Die Lage war sehr Zentral und man konnte alles sehenswertes zu Fuß erreichen. Wer am Wochenende nachts schlafen möchte, sollte diese Unterkunft auf keinen Fall nehmen. Party direkt vor der Tür so das man denkt, man schläft mitten drin. Sehr Sehr laut also und das bis früh 5 Uhr. Ab 7 kommt dann die Straßenreinigung die keineswegs leiser ist."),
             Document("Ótima escolha! Apartamento confortável e limpo! O RoofTop é otimo para beber um vinho! O apartamento é localizado entre duas ruas de movimento noturno. Porem as janelas, blindam 90% do barulho. Não nos incomodou"),
             Document("Nous avons passé un séjour formidable. Merci aux personnes , le bonjours à Ricardo notre taxi man, très sympathique. Je pense refaire un séjour parmi vous, après le confinement, tout était parfait, surtout leur gentillesse, aucune chaude négative. Je n'ai rien à redire de négative, Ils étaient a notre écoute, un gentil message tout les matins, pour nous demander si nous avions besoins de renseignement et savoir si tout allait bien pendant notre séjour."),
             Document("Boa localização. Bom pequeno almoço. A tv não se encontrava funcional."),
             Document("Céntrico. Muy cómodo para moverse y ver Oporto. Edificio con terraza propia en la última planta. Todo reformado y nuevo. Te traen un estupendo desayuno todas las mañanas al apartamento. Solo que se puede escuchar algo de ruido de la calle a primeras horas de la noche. Es un zona de ocio nocturno. Pero respetan los horarios.")
]

indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=preprocessor, name="preprocessor", inputs=["File"])
indexing_pipeline.add_node(component=retriever, name="retriever", inputs=["preprocessor"])
indexing_pipeline.add_node(component=document_store, name="document_store", inputs=['retriever'])
indexing_pipeline.run(documents=documents)


prompt="""
You will be provided with reviews in multiple languages for an accommodation.
Create a concise and informative answer for a given question based solely on the given reviews.
\nReviews: {join(documents)}
\nQuestion: {query};
\nAnswer:
"""
template = PromptTemplate(
    prompt=prompt,
    output_parser=AnswerParser())

prompt_node = PromptNode(model_name_or_path="command", api_key=COHERE_API_KEY, default_prompt_template=template)

rag_pipeline = Pipeline()
rag_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
rag_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
results = rag_pipeline.run("Is this place too noisy to sleep?", params={"Retriever": {"top_k": 3}})
logger.debug(results)

results = rag_pipeline.run("What are the problems about this place?", params={"Retriever": {"top_k": 10}})
logger.debug(results)

results = rag_pipeline.run("What is good about this place?", params={"Retriever": {"top_k": 10}})
logger.debug(results)

results = rag_pipeline.run("Should I stay at this hotel?", params={"Retriever": {"top_k": 10}})
logger.debug(results)

results = rag_pipeline.run("How is the wifi?", params={"Retriever": {"top_k": 3}})
logger.debug(results)

results = rag_pipeline.run("How can I use the coffee maker?", params={"Retriever": {"top_k": 3}})
logger.debug(results)

results = rag_pipeline.run("What are the attractions around this place?", params={"Retriever": {"top_k": 3}})
logger.debug(results)

results = rag_pipeline.run("Are there pubs near by?", params={"Retriever": {"top_k": 3}})
logger.debug(results)

logger.info("\n\n[DONE]", bright=True)