from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import logging
from typing import AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Streaming API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamRequest(BaseModel):
    message: str


async def stream_data(message: str) -> AsyncGenerator[str, None]:
    """Generate streaming data based on the input message."""
    logger.info(f"Starting stream for message: {message}")
    for i in range(5):
        chunk = f"Chunk {i+1}: {message}\n"
        logger.debug(f"Sending chunk: {chunk.strip()}")
        yield chunk
        await asyncio.sleep(1)  # Simulate processing delay
    logger.info("Stream completed")


@app.post("/api/stream")
async def stream_endpoint(request: StreamRequest) -> StreamingResponse:
    """
    POST endpoint for streaming response.
    Expects a JSON body with a 'message' field.
    """
    try:
        return StreamingResponse(
            stream_data(request.message),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Error in stream_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
