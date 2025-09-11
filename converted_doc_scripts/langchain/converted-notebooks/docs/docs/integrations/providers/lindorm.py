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
# Lindorm

Lindorm is a cloud-native multimodal database from Alibaba-Cloud, It supports unified access and integrated processing of various types of data, including wide tables, time-series, text, objects, streams, and spatial data. It is compatible with multiple standard interfaces such as SQL, HBase/Cassandra/S3, TSDB, HDFS, Solr, and Kafka, and seamlessly integrates with third-party ecosystem tools. This makes it suitable for scenarios such as logging, monitoring, billing, advertising, social networking, travel, and risk control. Lindorm is also one of the databases that support Alibaba's core businesses. 

To use the AI and vector capabilities of Lindorm, you should [get the service](https://help.aliyun.com/document_detail/174640.html?spm=a2c4g.11186623.help-menu-172543.d_0_1_0.4c6367558DN8Uq) and install `langchain-lindorm-integration` package.
"""
logger.info("# Lindorm")

# !
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
U

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
l
i
n
d
o
r
m
-
i
n
t
e
g
r
a
t
i
o
n

"""
## Embeddings

To use the embedding model deployed in Lindorm AI Service, import the LindormAIEmbeddings.
"""
logger.info("## Embeddings")

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
l
i
n
d
o
r
m
_
i
n
t
e
g
r
a
t
i
o
n

i
m
p
o
r
t

L
i
n
d
o
r
m
A
I
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

"""
## Rerank

The Lindorm AI Service also supports reranking.
"""
logger.info("## Rerank")

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
l
i
n
d
o
r
m
_
i
n
t
e
g
r
a
t
i
o
n
.
r
e
r
a
n
k
e
r

i
m
p
o
r
t

L
i
n
d
o
r
m
A
I
R
e
r
a
n
k

"""
## Vector Store

Lindorm also supports vector store.
"""
logger.info("## Vector Store")

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
l
i
n
d
o
r
m
_
i
n
t
e
g
r
a
t
i
o
n

i
m
p
o
r
t

L
i
n
d
o
r
m
V
e
c
t
o
r
S
t
o
r
e

"""
## ByteStore

Use ByteStore from Lindorm
"""
logger.info("## ByteStore")

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
l
i
n
d
o
r
m
_
i
n
t
e
g
r
a
t
i
o
n

i
m
p
o
r
t

L
i
n
d
o
r
m
B
y
t
e
S
t
o
r
e

logger.info("\n\n[DONE]", bright=True)