from PIL import Image
from datasets import load_dataset
from io import BytesIO
from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument
from llama_index.core.settings import Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from time import sleep
import base64
import matplotlib.pyplot as plt
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# **Multi-Modal Retrieval using VoyageAI Multi-Modal Embeddings**

VoyageAI has released a multi-modal embedding model and in this notebook, we will demonstrate Multi-Modal Retrieval using VoyageAI MultiModal Embeddings.

For the demonstration, here are the steps:

1. Download a dataset with images from HuggingFace.
2. Build a Multi-Modal index for images using VoyageAI Multi-Modal Embeddings.
3. Retrieve relevant images simultaneously using a Multi-Modal Retriever for a query.

# **Installation**

We will use VoyageAI MultiModal embeddings for retrieval and MongoDB as the vector-store.
"""
logger.info("# **Multi-Modal Retrieval using VoyageAI Multi-Modal Embeddings**")

# %pip install datasets
# %pip install llama-index
# %pip install llama-index-embeddings-voyageai
# %pip install llama-index-vector-stores-mongodb
# %pip install pymongo
# %pip install matplotlib



"""
# **Utils**

* plot_images: Plot the images in the specified list of image paths.
"""
logger.info("# **Utils**")

def plot_images(images, image_indexes):
    images_shown = 0
    for image_ndx in image_indexes:
        image = Image.open(BytesIO(images[image_ndx]))

        plt.subplot(8, 8, images_shown + 1)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

        images_shown += 1
        if images_shown >= 50:
            break

    plt.tight_layout()
    plt.show()

"""
# **Donwload the images**
We will download the dataset with the images.
"""
logger.info("# **Donwload the images**")

logger.debug("Loading dataset...")
dataset = load_dataset("princeton-nlp/CharXiv", split="validation")
df = dataset.to_pandas()

"""
Work with 50 images only.
"""
logger.info("Work with 50 images only.")

logger.debug("Processing images...")
image_bytes = []
for index, row in df.iterrows():
    image_data = row["image"]
    if image_data is not None:
        image_bytes.append(image_data["bytes"])
        if len(image_bytes) == 50:
            break

"""
Just show the images.
"""
logger.info("Just show the images.")

logger.debug("Showing images...")
plot_images(image_bytes, [x for x in range(len(image_bytes))])

"""
Now let's create documents, so we can then store these in the MongoDB database.
"""
logger.info("Now let's create documents, so we can then store these in the MongoDB database.")

logger.debug("Creting documents...")
documents = [
    ImageDocument(image=base64.b64encode(img), metadata={"index": ndx})
    for ndx, img in enumerate(image_bytes)
]

"""
Now, let's connect to MongoDB Atlas instance, define the image and text indexes and create the storage context. Also, initialise the multimodal embedding model.
"""
logger.info("Now, let's connect to MongoDB Atlas instance, define the image and text indexes and create the storage context. Also, initialise the multimodal embedding model.")

logger.debug("Setup...")
MONGO_URI = os.environ.get("MONGO_URI", "<YOUR_MONGODB_ATLAS_URL>")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "<YOUR_VOYAGE_API_KEY>")
db_name = "multimodal_test"
collection_name = "test"

client = MongoClient(MONGO_URI)

image_store = MongoDBAtlasVectorSearch(
    client,
    db_name=db_name,
    collection_name=f"{collection_name}_image",
    vector_index_name="image_vector_index",
)
image_store.create_vector_search_index(
    dimensions=1024, path="embedding", similarity="cosine"
)

text_store = MongoDBAtlasVectorSearch(
    client,
    db_name=db_name,
    collection_name=f"{collection_name}_text",
    vector_index_name="text_vector_index",
)
text_store.create_vector_search_index(
    dimensions=1024, path="embedding", similarity="cosine"
)

storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

Settings.embed_model = VoyageEmbedding(
    voyage_api_key=VOYAGE_API_KEY,
    model_name="voyage-multimodal-3",
    truncation=False,
)
Settings.chunk_size = 100
Settings.chunk_overlap = 10

"""
We can now store the images in MongoDB.
"""
logger.info("We can now store the images in MongoDB.")

logger.debug("Storing documents in MongoDB Atlas Vector Search...")
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
    image_embed_model=Settings.embed_model,
)

logger.debug("Finished storing images.")

"""
# **Test the Retrieval**
Here we create a retriever and test it out. Note that we are storing only images in our database, and we will query these images with plain text!
"""
logger.info("# **Test the Retrieval**")

retriever = index.as_retriever(similarity_top_k=2)

logger.debug("Performing query...")
nodes = retriever.text_to_image_retrieve(
    "3D loss landscapes for different training strategies"
)

"""
Inspect the retrieval results
"""
logger.info("Inspect the retrieval results")

logger.debug(f"Found {len(nodes)} results:")
result_images = []
for i, node in enumerate(nodes):
    ndx = node.metadata["index"]
    result_images.append(ndx)
plot_images(image_bytes, result_images)

logger.debug("Querying finished")

"""
We are done, so we can close the MongoDB connection.
"""
logger.info("We are done, so we can close the MongoDB connection.")

client.close()
logger.debug("MongoDB connection closed")

logger.info("\n\n[DONE]", bright=True)