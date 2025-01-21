import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sample_router import router
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Enable parallelism for faster LLM tokenizer encoding
os.environ["TOKENIZERS_PARALLELISM"] = "true"


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# app.middleware("http")(log_exceptions_middleware)

# Include the routes
# app.include_router(rag_router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(router)


# Run a simple test if this module is the main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
