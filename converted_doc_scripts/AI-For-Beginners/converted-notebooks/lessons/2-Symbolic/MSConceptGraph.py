from jet.logger import logger
from textblob import TextBlob
import json
import os
import shutil
import ssl
import sys
import urllib


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
## Microsoft Concept Graph

[Microsoft Concept Graph](https://concept.research.microsoft.com/) is a large taxonomy of terms mined from the internet, with `is-a` relations between concepts. 

Context Graph is available in two forms:
 * Large text file for download
 * REST API

Statistics:
 * 5401933 unique concepts, 
 * 12551613 unique instances
 * 87603947 `is-a` relations

## Using Web Service

Web service offers different calls to estimate probability of a concept belonging to different groups. More info is available [here](https://concept.research.microsoft.com/Home/Api).
Here is the sample URL to call: `https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance=microsoft&topK=10`
"""
logger.info("## Microsoft Concept Graph")


def http(x):
    ssl._create_default_https_context = ssl._create_unverified_context
    response = urllib.request.urlopen(x)
    data = response.read()
    return data.decode('utf-8')

def query(x):
    return json.loads(http("https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance={}&topK=10".format(urllib.parse.quote(x))))

query('microsoft')

"""
Let's try to categorize the news titles using parent concepts. To get news titles, we will use [NewsApi.org](http://newsapi.org) service. You need to obtain your own API key in order to use the service - go to the web site and register for free developer plan.
"""
logger.info("Let's try to categorize the news titles using parent concepts. To get news titles, we will use [NewsApi.org](http://newsapi.org) service. You need to obtain your own API key in order to use the service - go to the web site and register for free developer plan.")

news
def get_news(country='us'):
    res = json.loads(http("https://newsapi.org/v2/top-headlines?country={0}&apiKey={1}".format(country,newsapi_key)))
    return res['articles']

all_titles = [x['title'] for x in get_news('us')+get_news('gb')]

all_titles

"""
First of all, we want to be able to extract nouns from news titles. We will use `TextBlob` library to do this, which simplifies a lot of typical NLP tasks like this.
"""
logger.info("First of all, we want to be able to extract nouns from news titles. We will use `TextBlob` library to do this, which simplifies a lot of typical NLP tasks like this.")

# !{sys.executable} -m pip install textblob
# !{sys.executable} -m textblob.download_corpora

w = {}
for x in all_titles:
    for n in TextBlob(x).noun_phrases:
        if n in w:
            w[n].append(x)
        else:
            w[n]=[x]
{ x:len(w[x]) for x in w.keys()}

"""
We can see that nouns do not give us large thematic groups. Let's substitute nouns by more general terms obtained from the concept graph. This will take some time, because we are doing REST call for each noun phrase.
"""
logger.info("We can see that nouns do not give us large thematic groups. Let's substitute nouns by more general terms obtained from the concept graph. This will take some time, because we are doing REST call for each noun phrase.")

w = {}
for x in all_titles:
    for noun in TextBlob(x).noun_phrases:
        terms = query(noun.replace(' ','%20'))
        for term in [u for u in terms.keys() if terms[u]>0.1]:
            if term in w:
                w[term].append(x)
            else:
                w[term]=[x]

{ x:len(w[x]) for x in w.keys() if len(w[x])>3}

logger.debug('\nECONOMY:\n'+'\n'.join(w['economy']))
logger.debug('\nNATION:\n'+'\n'.join(w['nation']))
logger.debug('\nPERSON:\n'+'\n'.join(w['person']))

logger.info("\n\n[DONE]", bright=True)