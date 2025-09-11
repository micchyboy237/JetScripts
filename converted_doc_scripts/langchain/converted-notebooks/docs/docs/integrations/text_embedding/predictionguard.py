from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
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
#
 
P
r
e
d
i
c
t
i
o
n
G
u
a
r
d
E
m
b
e
d
d
i
n
g
s

>
[
P
r
e
d
i
c
t
i
o
n
 
G
u
a
r
d
]
(
h
t
t
p
s
:
/
/
p
r
e
d
i
c
t
i
o
n
g
u
a
r
d
.
c
o
m
)
 
i
s
 
a
 
s
e
c
u
r
e
,
 
s
c
a
l
a
b
l
e
 
G
e
n
A
I
 
p
l
a
t
f
o
r
m
 
t
h
a
t
 
s
a
f
e
g
u
a
r
d
s
 
s
e
n
s
i
t
i
v
e
 
d
a
t
a
,
 
p
r
e
v
e
n
t
s
 
c
o
m
m
o
n
 
A
I
 
m
a
l
f
u
n
c
t
i
o
n
s
,
 
a
n
d
 
r
u
n
s
 
o
n
 
a
f
f
o
r
d
a
b
l
e
 
h
a
r
d
w
a
r
e
.

#
#
 
O
v
e
r
v
i
e
w

### Integration details
This integration shows how to use the Prediction Guard embeddings integration with Langchain. This integration supports text and images, separately or together in matched pairs.

## Setup
To access Prediction Guard models, contact us [here](https://predictionguard.com/get-started) to get a Prediction Guard API key and get started.

### Credentials
Once you have a key, you can set it with
"""
logger.info("#")


os.environ["PREDICTIONGUARD_API_KEY"] = "<Prediction Guard API Key"

"""
#
#
#
 
I
n
s
t
a
l
l
a
t
i
o
n
"""
logger.info("#")

# %
p
i
p

i
n
s
t
a
l
l

-
-
u
p
g
r
a
d
e

-
-
q
u
i
e
t

l
a
n
g
c
h
a
i
n
-
p
r
e
d
i
c
t
i
o
n
g
u
a
r
d

"""
#
#
 
I
n
s
t
a
n
t
i
a
t
i
o
n

F
i
r
s
t
,
 
i
n
s
t
a
l
l
 
t
h
e
 
P
r
e
d
i
c
t
i
o
n
 
G
u
a
r
d
 
a
n
d
 
L
a
n
g
C
h
a
i
n
 
p
a
c
k
a
g
e
s
.
 
T
h
e
n
,
 
s
e
t
 
t
h
e
 
r
e
q
u
i
r
e
d
 
e
n
v
 
v
a
r
s
 
a
n
d
 
s
e
t
 
u
p
 
p
a
c
k
a
g
e
 
i
m
p
o
r
t
s
.
"""
logger.info("#")

f
r
o
m

l
a
n
g
c
h
a
i
n
_
p
r
e
d
i
c
t
i
o
n
g
u
a
r
d

i
m
p
o
r
t

P
r
e
d
i
c
t
i
o
n
G
u
a
r
d
E
m
b
e
d
d
i
n
g
s

embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")

"""


Prediction Guard embeddings generation supports both text and images. This integration includes that support spread across various functions.

#
#
 
I
n
d
e
x
i
n
g
 
a
n
d
 
R
e
t
r
i
e
v
a
l
"""
logger.info("#")


text = "LangChain is the framework for building context-aware reasoning applications."

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

retrieved_documents = retriever.invoke("What is LangChain?")

retrieved_documents[0].page_content

"""
## Direct Usage
The vectorstore and retriever implementations are calling `embeddings.embed_documents(...)` and `embeddings.embed_query(...)` to create embeddings from the texts used in the `from_texts` and retrieval `invoke` operations.

These methods can be directly called with the following commands.

#
#
#
 
E
m
b
e
d
 
s
i
n
g
l
e
 
t
e
x
t
s
"""
logger.info("## Direct Usage")

text = "This is an embedding example."
single_vector = embeddings.embed_query(text)

single_vector[:5]

"""
#
#
#
 
E
m
b
e
d
 
m
u
l
t
i
p
l
e
 
t
e
x
t
s
"""
logger.info("#")

docs = [
    "This is an embedding example.",
    "This is another embedding example.",
]

two_vectors = embeddings.embed_documents(docs)

for vector in two_vectors:
    logger.debug(vector[:5])

"""
#
#
#
 
E
m
b
e
d
 
s
i
n
g
l
e
 
i
m
a
g
e
s
"""
logger.info("#")

image = [
    "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg",
]
single_vector = embeddings.embed_images(image)

logger.debug(single_vector[0][:5])

"""
#
#
#
 
E
m
b
e
d
 
m
u
l
t
i
p
l
e
 
i
m
a
g
e
s
"""
logger.info("#")

images = [
    "https://fastly.picsum.photos/id/866/200/300.jpg?hmac=rcadCENKh4rD6MAp6V_ma-AyWv641M4iiOpe1RyFHeI",
    "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg",
]

two_vectors = embeddings.embed_images(images)

for vector in two_vectors:
    logger.debug(vector[:5])

"""
#
#
#
 
E
m
b
e
d
 
s
i
n
g
l
e
 
t
e
x
t
-
i
m
a
g
e
 
p
a
i
r
s
"""
logger.info("#")

inputs = [
    {
        "text": "This is an embedding example.",
        "image": "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg",
    },
]
single_vector = embeddings.embed_image_text(inputs)

logger.debug(single_vector[0][:5])

"""
#
#
#
 
E
m
b
e
d
 
m
u
l
t
i
p
l
e
 
t
e
x
t
-
i
m
a
g
e
 
p
a
i
r
s
"""
logger.info("#")

inputs = [
    {
        "text": "This is an embedding example.",
        "image": "https://fastly.picsum.photos/id/866/200/300.jpg?hmac=rcadCENKh4rD6MAp6V_ma-AyWv641M4iiOpe1RyFHeI",
    },
    {
        "text": "This is another embedding example.",
        "image": "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg",
    },
]
two_vectors = embeddings.embed_image_text(inputs)

for vector in two_vectors:
    logger.debug(vector[:5])

"""
## API Reference
For detailed documentation of all PredictionGuardEmbeddings features and configurations check out the API reference: https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.predictionguard.PredictionGuardEmbeddings.html
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)