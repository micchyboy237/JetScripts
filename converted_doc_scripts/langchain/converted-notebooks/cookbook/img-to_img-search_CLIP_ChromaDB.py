from PIL import Image
from jet.logger import logger
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
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
# Image to Image search Using Ollama's Open source CLIP Model (Based on Vision Transformer) and ChromaDB

#### This Cookbook demonstrates A reverse image search or image similarity search, using an input image and some provided images which will be indexed or embedded in ChromaDB

#
#
#
#
#
 
Y
o
u
 
c
a
n
 
e
m
b
e
d
 
t
e
x
t
 
i
n
 
t
h
e
 
s
a
m
e
 
V
e
c
t
o
r
D
B
 
s
p
a
c
e
 
a
s
 
i
m
a
g
e
s
,
 
a
n
d
 
r
e
t
r
i
e
v
e
 
t
e
x
t
 
a
n
d
 
i
m
a
g
e
s
 
a
s
 
w
e
l
l
 
b
a
s
e
d
 
o
n
 
i
n
p
u
t
 
t
e
x
t
 
o
r
 
i
m
a
g
e
.

#
#
#
#
#
 
F
o
l
l
o
w
i
n
g
 
l
i
n
k
 
d
e
m
o
n
s
t
r
a
t
e
s
 
t
h
a
t
.

<
a
>
 
h
t
t
p
s
:
/
/
p
y
t
h
o
n
.
l
a
n
g
c
h
a
i
n
.
c
o
m
/
v
0
.
2
/
d
o
c
s
/
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
s
/
t
e
x
t
_
e
m
b
e
d
d
i
n
g
/
o
p
e
n
_
c
l
i
p
/
 
<
/
a
>

## Installs and imports
"""
logger.info("# Image to Image search Using Ollama's Open source CLIP Model (Based on Vision Transformer) and ChromaDB")

# !pip install langchain_experimental

# !pip install langchain_chroma



"""
### Langchain Imports
"""
logger.info("### Langchain Imports")


"""
## Provide your paths in a list

#### This Cookbook uses data from this Myntra Kaggle dataset :- <a> https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset </a>
#### You can directly download images or read the csv and links from it and then download
"""
logger.info("## Provide your paths in a list")

all_image_uris = [
    "../../../py_ml_env/images_all/b0eb9426-adf2-4802-a6b3-5dbacbc5f2511643971561167KhushalKWomenBlackEthnicMotifsAngrakhaBeadsandStonesKurtawit7.jpg",
    "../../../py_ml_env/images_all/17ab2ac8-2e60-422d-9d20-2527415932361640754214931-STRAPPY-SET-IN-ORANGE-WITH-ORGANZA-DUPATTA-5961640754214349-2.jpg",
    "../../../py_ml_env/images_all/b8c4f90f-683c-48d2-b8ac-19891a87c0651638428628378KurtaSets1.jpg",
    "../../../py_ml_env/images_all/d2407657-1f04-4d13-9f52-9e134050489b1625905793495-Nayo-Women-Red-Ethnic-Motifs-Printed-Empire-Pure-Cotton-Kurt-1.jpg",
    "../../../py_ml_env/images_all/30b0017d-7e72-4d40-9633-ef78d01719741575541717470-AHIKA-Women-Black--Green-Printed-Straight-Kurta-990157554171-1.jpg",
    "../../../py_ml_env/images_all/507490f7-c8f9-492c-b3f8-c7e977d1af701654922515416SochWomenRedThreadWorkGeorgetteAnarkaliKurta1.jpg",
    "../../../py_ml_env/images_all/5fba9594-3301-4881-ba56-d56a44570e831654747998773LibasWomenNavyBluePureCottonFloralPrintKurtawithPalazzosDupa1.jpg",
    "../../../py_ml_env/images_all/e6b90907-a613-45e1-9b2e-988caaba36581645010770505-Ahalyaa-Women-Beige-Floral-Printed-Regular-Gotta-Patti-Kurta-1.jpg",
    "../../../py_ml_env/images_all/5ea707f4-8491-4d1c-b520-86a1cff4c86e1644841891629-Anouk-Women-Yellow--White-Printed-Kurta-with-Palazzos-706164-1.jpg",
    "../../../py_ml_env/images_all/11b842c5-d9d4-4fee-baa2-0972e3a673641643970773675KhushalKWomenGreenEthnicMotifsPrintedEmpireGottaPattiPureCot7.jpg",
    "../../../py_ml_env/images_all/b783aef9-c902-462e-af73-de159bfd011c1565256752191-Libas-Women-Kurta-Sets-2081565256750830-1.jpg",
    "../../../py_ml_env/images_all/bb925efb-80d9-4cb6-838c-df86f1ba3c3e1637570416652-Varanga-Women-Mustard-Yellow-Floral-Yoke-Embroidered-Straigh-1.jpg",
    "../../../py_ml_env/images_all/7d7656e5-e37d-4f61-9407-98bd341ca8f91640261029836KurtaSets1.jpg",
    "../../../py_ml_env/images_all/43d65352-9853-498e-95a4-be514df0be901559294212152-Vishudh--Straight-Kurta-With-Crop-Palazzo-7041559294209627-1.jpg",
    "../../../py_ml_env/images_all/4a37718e-8942-479c-a7ea-0b074d53ee4b1650456566424AnoukWomenPeach-ColouredYokeDesignMirror-WorkKurtawithTrouse1.jpg",
    "../../../py_ml_env/images_all/5910af54-3435-40d5-95d4-0ac2daf797f51658319613886-SheWill-Women-Maroon-Ethnic-Yoke-Design-Embroided-Kurta-with-1.jpg",
    "../../../py_ml_env/images_all/d57adb8b-e792-477a-8801-6ea570cd88ef1629800170287VarangaWomenYellowFloralPrintedKeyholeNeckThreadWorkKurta1.jpg",
    "../../../py_ml_env/images_all/c35d059d-a357-4863-bcb1-eacd8c988fb01572422803188-AHIKA-Women-Kurtas-8841572422802083-1.jpg",
    "../../../py_ml_env/images_all/3a61f2ab-7905-4efc-84e8-df1f74fa08201623409397327-Anouk-Women-Kurtas-1031623409396642-1.jpg",
    "../../../py_ml_env/images_all/3e9c355b-20e6-42d0-8480-7046979f87711658733247220CharuWomenNavyBlueStripedThreadWorkKurta1.jpg",
    "../../../py_ml_env/images_all/0d391a8b-ea8c-4258-86d5-a99b9f3f34201630040200642-Libas-Women-Kurta-Sets-5941630040199555-1.jpg",
    "../../../py_ml_env/images_all/d6b74d2b-825f-4b34-af01-9d6336045bdb1624612149604-1.jpg",
    "../../../py_ml_env/images_all/07adcdf7-eee1-4077-b55c-f6608caaa6f01647663614971KALINIWomenSeaGreenFloralYokeDesignPleatedPureCottonTopwithS4.jpg",
    "../../../py_ml_env/images_all/6bc412bb-3cc6-4def-8833-f5580b0cc06a1617706648250-Indo-Era-Green-Printed-Straight-Kurta-Palazzo-With-Dupatta-S-1.jpg",
    "../../../py_ml_env/images_all/b1bd0687-7533-428d-8258-d29c793fc4541631092430795-Anouk-Women-Kurta-Sets-941631092429795-1.jpg",
    "../../../py_ml_env/images_all/64e975d5-dbda-4c09-87c0-c5152f9e82c71658736715566TOULINWomenTealFloralAngrakhaKurtiwithPalazzosWithDupatta1.jpg",
    "../../../py_ml_env/images_all/d1a4cc48-ff90-47ab-ad36-800743e83d641605767381033-Ishin-Womens-Rayon-Red-Bandhani-Print-Embellished-Anarkali-K-1.jpg",
]

"""
## (Optional) Prepare Metadata to index alongside the image
"""
logger.info("## (Optional) Prepare Metadata to index alongside the image")

metadatas = []
for idx, img in enumerate(all_image_uris):
    meta_dict = {}
    meta_dict["path"] = img
    meta_dict["id"] = idx
    metadatas.append(meta_dict)
logger.debug(metadatas[:5])

"""
## Initialize the Ollama CLIP Model
"""
logger.info("## Initialize the Ollama CLIP Model")

model_name = "ViT-B-32"
checkpoint = "laion2b_s34b_b79k"

clip_embd = OpenCLIPEmbeddings(model_name=model_name, checkpoint=checkpoint)

"""
### Sample test of images
"""
logger.info("### Sample test of images")

img_feat_1 = clip_embd.embed_image([all_image_uris[0]])

"""
### Dimentions of embeddings
"""
logger.info("### Dimentions of embeddings")

len(img_feat_1[0])

"""
### Initialize the Chroma Client, persist_directory is optinal if you want to save the VectorDB to disk and reload it using same code and path
"""
logger.info("### Initialize the Chroma Client, persist_directory is optinal if you want to save the VectorDB to disk and reload it using same code and path")

collection_name = "chroma_img_collection_1"
chroma_client = Chroma(
    collection_name=collection_name,
    embedding_function=clip_embd,
    persist_directory="./indexed_db",
)

def embed_images(chroma_client, uris, metadatas=[]):
    """
    Function to add images to Chroma client with progress bar.

    Args:
        chroma_client: The Chroma client object.
        uris (List[str]): List of image file paths.
        metadatas (List[dict]): List of metadata dictionaries.
    """
    success_count = 0
    for i in tqdm(range(len(uris)), desc="Adding images"):
        uri = uris[i]
        metadata = metadatas[i]

        try:
            chroma_client.add_images(uris=[uri], metadatas=[metadata])
        except Exception as e:
            logger.debug(f"Failed to add image {uri} with metadata {metadata}. Error: {e}")
        else:
            success_count += 1

    return success_count

"""
### Specify your image paths list in this embed_images function call
"""
logger.info("### Specify your image paths list in this embed_images function call")

success_count = embed_images(chroma_client, uris=all_image_uris, metadatas=metadatas)
if success_count:
    logger.debug(f"{success_count} Images Embedded Successfully")
else:
    logger.debug("No images Embedded")

"""
## Helper function to plot retrieved similar images
"""
logger.info("## Helper function to plot retrieved similar images")




def plot_images_by_side(image_data):
    num_images = len(image_data)
    n_col = 2  # Fixed number of columns
    n_row = math.ceil(num_images / n_col)  # Calculate the number of rows

    fig, axs = plt.subplots(n_row, n_col, figsize=(10, 5 * n_row))
    axs = axs.flatten()

    for idx, data in enumerate(image_data):
        img_path = data["path"]
        score = round(data.get("score", 0), 2)
        img = Image.open(img_path)
        ax = axs[idx]
        ax.imshow(img)
        ax.title.set_text(f"\nProduct ID: {data['id']}\n Score: {score}")
        ax.axis("off")  # Turn off axis

    for i in range(num_images, n_row * n_col):
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()

"""
## Take in input image path, resize that image and display it
"""
logger.info("## Take in input image path, resize that image and display it")

search_img_path = "../../../py_ml_env/images_all/0d391a8b-ea8c-4258-86d5-a99b9f3f34201630040200642-Libas-Women-Kurta-Sets-5941630040199555-1.jpg"

my_image = Image.open(search_img_path).convert("RGB")
max_width = 400
max_height = 400

width, height = my_image.size
aspect_ratio = width / height

if width > height:
    new_width = min(width, max_width)
    new_height = int(new_width / aspect_ratio)
else:
    new_height = min(height, max_height)
    new_width = int(new_height * aspect_ratio)

my_image_resized = my_image.resize((new_width, new_height), Image.LANCZOS)

my_image_resized

"""
## Perform Image similarity search, get the metadata of K retrieved images and then display similar images

### We have embeded limited data, we can embed a large number which will have similar images, to get better results
"""
logger.info("## Perform Image similarity search, get the metadata of K retrieved images and then display similar images")

k = 10


similar_images = chroma_client.similarity_search_by_image(uri=search_img_path, k=k)

similar_image_data_1 = []
for img in similar_images:
    similar_image_data_1.append(img.metadata)
plot_images_by_side(similar_image_data_1)

"""
## Perform similarity search with image with relevance scores:
 We get a list of K tuples like following:
 [
    (Langchain_Document,score),
   (Langchain_Document,score),
   Langchain_Document,score)
   ]
"""
logger.info("## Perform similarity search with image with relevance scores:")

similar_images = chroma_client.similarity_search_by_image_with_relevance_score(
    uri=search_img_path, k=k
)

similar_image_data_2 = []
for img in similar_images:
    meta_dict = img[0].metadata
    meta_dict["score"] = img[1]
    similar_image_data_2.append(meta_dict)

plot_images_by_side(similar_image_data_2)

"""
## We have successfully implemented an image-to-image search using CLIP and ChromaDB !
"""
logger.info("## We have successfully implemented an image-to-image search using CLIP and ChromaDB !")

logger.info("\n\n[DONE]", bright=True)