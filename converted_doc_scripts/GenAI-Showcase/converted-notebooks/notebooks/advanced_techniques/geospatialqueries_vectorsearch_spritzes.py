from jet.logger import CustomLogger
from pymongo import MongoClient
import googlemaps
import ollama
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/advanced_techniques/geospatialqueries_vectorsearch_spritzes.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/mongodb/geospatial-queries-vector-search/)


First install your Google Maps library and install Ollama, since we will need the Google Maps library for our Google Places API and we will need Ollama to embed our documents.
"""
logger.info("First install your Google Maps library and install Ollama, since we will need the Google Maps library for our Google Places API and we will need Ollama to embed our documents.")

# !pip install googlemaps
# !pip install ollama==0.28

"""
# Let's now pass in our imports. We are going to be including the `getpass` library since we will need it to write in our secret keys.
"""
# logger.info("Let's now pass in our imports. We are going to be including the `getpass` library since we will need it to write in our secret keys.")

# import getpass


"""
Write in your secret keys, we will need to write in our Google API Key and our [Ollama API Key](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key).
"""
logger.info("Write in your secret keys, we will need to write in our Google API Key and our [Ollama API Key](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key).")

# google_api_key = getpass.getpass(prompt="Put in Google API Key here")
map_client = googlemaps.Client(key=google_api_key)

# ollama_api_key = getpass.getpass(prompt="Put in Ollama API Key here")

"""
Now, let's set ourselves up for Vector Search success. First, set your key and then establish our embedding function. For this tutorial, we are using Ollama's "mxbai-embed-large" embedding model. We are going to be embedding the reviews of our spritz locations so we can make some judgements on where to go!
"""
logger.info("Now, let's set ourselves up for Vector Search success. First, set your key and then establish our embedding function. For this tutorial, we are using Ollama's "mxbai-embed-large" embedding model. We are going to be embedding the reviews of our spritz locations so we can make some judgements on where to go!")

ollama.api_key = ollama_api_key

EMBEDDING_MODEL = "mxbai-embed-large"


def get_embedding(text):
    response = ollama.Embedding.create(input=text, model=EMBEDDING_MODEL)
    return response["data"][0]["embedding"]

"""
When using Nearby Search in our Google Maps API, we are required to set up three parameters: location, radius, and keyword. For our location, we can find our starting coordinates (the very middle of West Village) by right clicking on Google Maps and copying the coordinates to our clipboard. That is how I got the coordinates shown below.

For our radius, we have to have it in meters. Since I'm not very savvy with meters, let's write a small function to help us make that conversion.

Our keyword will just be what we're hoping to find from the Google Places API, aperol spritzes!

We can then make our API call using the `places_nearby` method.
"""
logger.info("When using Nearby Search in our Google Maps API, we are required to set up three parameters: location, radius, and keyword. For our location, we can find our starting coordinates (the very middle of West Village) by right clicking on Google Maps and copying the coordinates to our clipboard. That is how I got the coordinates shown below.")

def miles_to_meters(miles):
    return miles * 1609.344


middle_of_west_village = (40.73490473393682, -74.00521094160642)
search_radius = miles_to_meters(
    0.4
)  # West Village is small so just do less than half a mile.
spritz_finder = "aperol spritz"

response = map_client.places_nearby(
    location=middle_of_west_village, radius=search_radius, keyword=spritz_finder
)

"""
Before we can go ahead and print out our locations, let's think about our end goal. We want to achieve a couple of things before we insert our documents into our MongoDB Atlas cluster. We want to:
1. Get detailed information about our locations, so we need to make another API call to get our `place_id`, the location `name`, our `formatted_address`, the `geometry`, some `reviews` (only up to 5), and the location `rating`. You can find more fields to return (if your heart desires!) from the [Nearby Search documentation](https://developers.google.com/maps/documentation/places/web-service/search-nearby)

2. Then, we want to embed our reviews for each location using our embedding function. We want to make sure that we have a field for these so our vectors are stored in an array inside of our cluster. We are choosing to embed here just to make things easier for ourselves in the long run. Let's also join all the five reviews together into one string to make things a bit easier on ourselves with the embedding.

3. While we're creating our dictionary with all the important information we want to portray, we need to think about how our coordinates are set up. MongoDB Geospatial Queries requires GeoJSON format. This means we need to make sure we have the proper format, or else we won't be able to use our Geospatial Queries operators later. We also need to keep in mind that the longitude and latitude is stored in a nested array underneath geometry and location inside of our Google Places API. So, we unfortunatly can't just access it out, we need to work some magic first. Here is an example output of what I copied from the documentation:
```
{
  "html_attributions": [],
  "results":
    [
      {
        "business_status": "OPERATIONAL",
        "geometry":
          {
            "location": { "lat": -33.8587323, "lng": 151.2100055 },
            "viewport":
              {
                "northeast":
                  { "lat": -33.85739847010727, "lng": 151.2112436298927 },
                "southwest":
                  { "lat": -33.86009812989271, "lng": 151.2085439701072 },
              },

```

With all this in mind, let's get to it:
"""
logger.info("Before we can go ahead and print out our locations, let's think about our end goal. We want to achieve a couple of things before we insert our documents into our MongoDB Atlas cluster. We want to:")

spritz_locations = []
for location in response.get("results", []):
    location_detail = map_client.place(
        place_id=location["place_id"],
        fields=["name", "formatted_address", "geometry", "reviews", "rating"],
    )

    details = location_detail.get("result", {})

    location_reviews = details.get("reviews", [])
    store_reviews = [review["text"] for review in location_reviews[:5]]
    joined_reviews = " ".join(store_reviews)

    embedding_reviews = get_embedding(joined_reviews)

    geometry = details.get("geometry", {})
    location = geometry.get("location", {})

    longitute = location.get("lng")
    latitude = location.get("lat")

    location_info = {
        "name": details.get("name"),
        "address": details.get("formatted_address"),
        "location": {"type": "Point", "coordinates": [longitute, latitude]},
        "rating": details.get("rating"),
        "reviews": store_reviews,
        "embedding": embedding_reviews,
    }
    spritz_locations.append(location_info)

"""
Let's print out our output and see what our spritz locations in the West Village neighborhood of NYC are! Let's also check and make sure we can see an embedding field.
"""
logger.info("Let's print out our output and see what our spritz locations in the West Village neighborhood of NYC are! Let's also check and make sure we can see an embedding field.")

for location in spritz_locations:
    logger.debug(
        f"Name: {location['name']}, Address: {location['address']}, Coordinates: {location['location']}, Rating: {location['rating']}, Reviews: {location['reviews']}, Embedding: {location['embedding']}"
    )

"""
Now that we have our documents formatted the way we want them to be, let's insert everything into MongoDB Atlas using the `pymongo` library.

First, let's install `pymongo`
"""
logger.info("Now that we have our documents formatted the way we want them to be, let's insert everything into MongoDB Atlas using the `pymongo` library.")

# !pip install pymongo

"""
Now, let's set up our MongoDB Connection, in order to do this please make sure you have your connection string, if you need help finding it please refer to the documentation.

Keep in mind that you can name your database and collection anything you like! I am naming my database "spritz_summer" and my collection "spritz_locations_WV". Run the code block below to insert your documents into your cluster.
"""
logger.info("Now, let's set up our MongoDB Connection, in order to do this please make sure you have your connection string, if you need help finding it please refer to the documentation.")


# connection_string = getpass.getpass(
    prompt="Enter connection string WITH USER + PASS here"
)
client = MongoClient(
    connection_string, appname="devrel.showcase.geospatial_vector_search"
)

database = client["spritz_summer"]
collection = database["spritz_locations_WV"]

collection.insert_many(spritz_locations)

"""
Perfect! Go ahead and check back in MongoDB Atlas in your cluster and make sure that everything looks the way we want it to look before we proceed. Please double check that your embedding field is there and that it's an array of 1536.

## Which one comes first, Vector Search or Geospatial Queries?
Both of these need to be the first stage in their aggregation pipelines, so instead of making one pipeline we are going to do a little loophole. We will do two pipelines. But how will we decide which?!

When I'm using Google Maps to figure out where to go, I normally first search for what I'm looking for and then I see how far away it is from where I currently am and pick the closest location to me. So let's keep that mindset in place and start off with MongoDB Atlas Vector Search for this tutorial. But, I understand intuitively some of you might prefer to search via all nearby locations and then utilize Vector Search, so I'll highlight that method of searching for your spritz's as well.

## MongoDB Atlas Vector Search
We have a couple steps here. Our first step is to create a Vector Search Index. Do this inside of MongoDB Atlas by following this documentation. Please keep in mind that your index is NOT run in your script, it lives in your cluster. You'll know it's ready to go when the button turns green and it's activated.
"""
logger.info("## Which one comes first, Vector Search or Geospatial Queries?")

{
    "fields": [
        {
            "numDimensions": 1536,
            "path": "embedding",
            "similarity": "cosine",
            "type": "vector",
        }
    ]
}

"""
Once it's activated, let's get to Vector Searching!

So, let's say I just finished dinner with my besties at our favorite restaurant in the West Village, Balaboosta. The food was great and it's a warm summer day and we're in the mood for post dinner spritz's outside, and we would prefer to be seated quickly. Let's see if we can find a spot!

Our first step with building our our pipeline is to embed our query. We can't compare text to vectors, we have to compare vectors to vectors. Do this with only a couple lines since we are using the same embedding model that we embedded our reviews with:
"""
logger.info("Once it's activated, let's get to Vector Searching!")

query_description = "outdoor seating quick service"


query_vector = get_embedding(query_description)

"""
Now, let's build out our aggregation pipeline. Since we are going to be using a $geoNear pipeline next, we want to keep in the IDs found from this search:
"""
logger.info("Now, let's build out our aggregation pipeline. Since we are going to be using a $geoNear pipeline next, we want to keep in the IDs found from this search:")

spritz_near_me_vector = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": 15,
            "limit": 5,
        }
    },
    {
        "$project": {
            "_id": 1,  # we want to keep this in place so we can search again using GeoNear
            "name": 1,
            "rating": 1,
            "reviews": 1,
        }
    },
]

"""
Let's print out our results and see what happens from our query of "outdoor seating quick service" :
"""
logger.info("Let's print out our results and see what happens from our query of "outdoor seating quick service" :")

spritz_near_me_vector_results = list(collection.aggregate(spritz_near_me_vector))
for result in spritz_near_me_vector_results:
    logger.debug(result)

"""
We have five fantastic options! Let's go ahead and save the IDs from our above pipeline in a simple line:
"""
logger.info("We have five fantastic options! Let's go ahead and save the IDs from our above pipeline in a simple line:")

spritz_near_me_ids = [result["_id"] for result in spritz_near_me_vector_results]
logger.debug(spritz_near_me_ids)

"""
Now that they're saved, we can build out our $geoNear pipeline and see which one of these options is closest to us from our starting location, Balaboosta, and which one we can walk over to!

To figure out the coordinates of Balaboosta, I right clicked on Google Maps and saved in the coordinates and then made sure I was including the longitude and latitude in the proper order.
"""
logger.info("Now that they're saved, we can build out our $geoNear pipeline and see which one of these options is closest to us from our starting location, Balaboosta, and which one we can walk over to!")

collection.create_index({"location": "2dsphere"})

spritz_near_me_geo = [
    {
        "$geoNear": {
            "near": {
                "type": "Point",
                "coordinates": [-74.0059456749148, 40.73781277366724],
            },
            "query": {"_id": {"$in": spritz_near_me_ids}},
            "minDistance": 100,
            "maxDistance": 1000,
            "spherical": True,
            "distanceField": "dist.calculated",
        }
    },
    {
        "$project": {
            "_id": 0,
            "name": 1,
            "address": 1,
            "rating": 1,
            "dist.calculated": 1,
        }
    },
    {"$limit": 3},
    {"$sort": {"dist.calculated": 1}},
]

spritz_near_me_geo_results = collection.aggregate(spritz_near_me_geo)
for result in spritz_near_me_geo_results:
    logger.debug(result)

"""
Seems like the restaurant we're heading over to is Pastis since it's the closest and fits our criteria perfectly.

## Other way around! Geospatial Queries first, then Vector Search
"""
logger.info("## Other way around! Geospatial Queries first, then Vector Search")

collection.create_index({"location": "2dsphere"})

spritz_near_me_geo = [
    {
        "$geoNear": {
            "near": {
                "type": "Point",
                "coordinates": [-74.0059456749148, 40.73781277366724],
            },
            "minDistance": 100,
            "maxDistance": 1000,
            "spherical": True,
            "distanceField": "dist.calculated",
        }
    },
    {"$project": {"_id": 1, "dist.calculated": 1}},
]

places_ids = list(collection.aggregate(spritz_near_me_geo))
distances = {
    result["_id"]: result["dist"]["calculated"] for result in places_ids
}  # have to create a new dictionary to keep our distances
spritz_near_me_ids = [result["_id"] for result in places_ids]

vector_search_index = {
    "fields": [
        {
            "numDimensions": 1536,
            "path": "embedding",
            "similarity": "cosine",
            "type": "vector",
        },
        {"type": "filter", "path": "_id"},
    ]
}

spritz_near_me_vector = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": 15,
            "limit": 3,
            "filter": {"_id": {"$in": spritz_near_me_ids}},
        }
    },
    {
        "$project": {
            "_id": 1,  # we want to keep this in place
            "name": 1,
            "rating": 1,
            "dist.calculated": 1,
        }
    },
]


spritz_near_me_vector_results = collection.aggregate(spritz_near_me_vector)
for result in spritz_near_me_vector_results:
    result["dist.calculated"] = distances.get(result["_id"])
    logger.debug(result)

logger.info("\n\n[DONE]", bright=True)