from fastapi import FastAPI

# The file where NeuralSearcher is stored
from neural_searcher import NeuralSearcher

app = FastAPI()

# Create a neural searcher instance
neural_searcher = NeuralSearcher(collection_name="startups")


@app.get("/api/search")
def search_startup(q: str):
    result = neural_searcher.search(text=q)
    return {"result": result}


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)
