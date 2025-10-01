from bertopic import BERTopic
from jet.logger import logger
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

"""
When using HDBSCAN, DBSCAN, or OPTICS, a number of outlier documents might be created
that do not fall within any of the created topics. These are labeled as -1. Depending on your use case, you might want
to decrease the number of documents that are labeled as outliers. Fortunately, there are a number of strategies one might
use to reduce the number of outliers after you have trained your BERTopic model.

The main way to reduce your outliers in BERTopic is by using the `.reduce_outliers` function. To make it work without too much tweaking, you will only need to pass the `docs` and their corresponding `topics`. You can pass outlier and non-outlier documents together since it will only try to reduce outlier documents and label them to a non-outlier topic.

The following is a minimal example:
"""
logger.info("When using HDBSCAN, DBSCAN, or OPTICS, a number of outlier documents might be created")


topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

new_topics = topic_model.reduce_outliers(docs, topics)

"""
!!! note
    You can use the `threshold` parameter to select the minimum distance or similarity when matching outlier documents with non-outlier topics. This allows the user to change the amount of outlier documents are assigned to non-outlier topics.


## **Strategies**

The default method for reducing outliers is by calculating the c-TF-IDF representations of outlier documents and assigning them
to the best matching c-TF-IDF representations of non-outlier topics.

However, there are a number of other strategies one can use, either separately or in conjunction that are worthwhile to explore:

* Using the topic-document probabilities to assign topics
* Using the topic-document distributions to assign topics
* Using c-TF-IDF representations to assign topics
* Using document and topic embeddings to assign topics

### **Probabilities**
This strategy uses the soft-clustering as performed by HDBSCAN to find the
best matching topic for each outlier document. To use this, make
sure to calculate the `probabilities` beforehand by instantiating
BERTopic with `calculate_probabilities=True`.
"""
logger.info("## **Strategies**")


topic_model = BERTopic(calculate_probabilities=True)
topics, probs = topic_model.fit_transform(docs)

new_topics = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities")

"""
### **Topic Distributions**
Use the topic distributions, as calculated with `.approximate_distribution`
to find the most frequent topic in each outlier document. You can use the
`distributions_params` variable to tweak the parameters of
`.approximate_distribution`.
"""
logger.info("### **Topic Distributions**")


topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

new_topics = topic_model.reduce_outliers(docs, topics, strategy="distributions")

"""
### **c-TF-IDF**
Calculate the c-TF-IDF representation for each outlier document and
find the best matching c-TF-IDF topic representation using
cosine similarity.
"""
logger.info("### **c-TF-IDF**")


topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

new_topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf")

"""
### **Embeddings**
Using the embeddings of each outlier documents, find the best
matching topic embedding using cosine similarity.
"""
logger.info("### **Embeddings**")


topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings")

"""
!!! note
    If you have pre-calculated the documents embeddings you can speed up the outlier
    reduction process for the `"embeddings"` strategy as it will prevent re-calculating
    the document embeddings.

### **Chain Strategies**

Since the `.reduce_outliers` function does not internally update the topics, we can easily try out different strategies but also chain them together.
You might want to do a first pass with the `"c-tf-idf"` strategy as it is quite fast. Then, we can perform the `"distributions"` strategy on the
outliers that are left since this method is typically much slower:
"""
logger.info("### **Chain Strategies**")

new_topics = topic_model.reduce_outliers(docs, topics , strategy="c-tf-idf", threshold=0.1)

new_topics = topic_model.reduce_outliers(docs, new_topics, strategy="distributions")

"""
## **Update Topics**

After generating our updated topics, we can feed them back into BERTopic in one of two ways. We can either update the topic representations themselves based on the documents that now belong to new topics or we can only update the topic frequency without updating the topic representations themselves.

!!! warning
    In both cases, it is important to realize that
    updating the topics this way may lead to errors if topic reduction or topic merging techniques are used afterwards. The reason for this is that when you assign a -1 document to topic 1 and another -1 document to topic 2, it is unclear how you map the -1 documents. Is it matched to topic 1 or 2.


### **Update Topic Representation**

When outlier documents are generated, they are not used when modeling the topic representations. These documents are completely ignored when finding good descriptions of topics. Thus, after having reduced the number of outliers in your topic model, you might want to update the topic representations with the documents that now belong to actual topics. To do so, we can make use of the `.update_topics` function:
"""
logger.info("## **Update Topics**")

topic_model.update_topics(docs, topics=new_topics)

"""
As seen above, you will only need to pass the documents on which the model was trained including the new topics that were generated using one of the above four strategies.

### **Exploration**

When you are reducing the number of topics, it might be worthwhile to iteratively visualize the results in order to get an intuitive understanding of the effect of the above four strategies. Making use of `.visualize_documents`, we can quickly iterate over the different strategies and view their effects. Here, an example will be shown on how to approach such a pipeline.

First, we train our model:
"""
logger.info("### **Exploration**")


docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
vectorizer_model = CountVectorizer(stop_words="english")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

reduced_embeddings = UMAP(n_neighbors=10, n_components=2,
                          min_dist=0.0, metric='cosine').fit_transform(embeddings)

topic_model = BERTopic(embedding_model=sentence_model, umap_model=umap_model,
                       vectorizer_model=vectorizer_model, calculate_probabilities=True, nr_topics=40)
topics, probs = topic_model.fit_transform(docs, embeddings)

"""
After having trained our model, let us take a look at the 2D representation of the generated topics:
"""
logger.info("After having trained our model, let us take a look at the 2D representation of the generated topics:")

topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings,
                                hide_document_hover=True, hide_annotations=True)

"""
<iframe src="fig_base.html" style="width:800px; height: 800px; border: 0px;""></iframe>


Next, we reduce the number of outliers using the `probabilities` strategy:
"""
logger.info("Next, we reduce the number of outliers using the `probabilities` strategy:")

new_topics = reduce_outliers(topic_model, docs, topics, probabilities=probs,
                             threshold=0.05, strategy="probabilities")
topic_model.update_topics(docs, topics=new_topics)

"""
And finally, we visualize the results:
"""
logger.info("And finally, we visualize the results:")

topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings,
                                hide_document_hover=True, hide_annotations=True)

"""
<iframe src="fig_reduced.html" style="width:800px; height: 800px; border: 0px;""></iframe>
"""

logger.info("\n\n[DONE]", bright=True)