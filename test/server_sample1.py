from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


def event_stream():
    """Generator function to yield events for streaming."""
    for i in range(10):  # Example: Stream 10 chunks
        yield f"data: Message {i}\n\n"
        time.sleep(1)  # Simulate delay between chunks


@app.post("/api/threads/{thread_id}/runs/stream")
async def stream_response(thread_id: str, request: Request):
    """
    Endpoint to stream responses for a given thread_id.
    """
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(event_stream(), headers=headers)

if __name__ == "__main__":
    import os
    import uvicorn

    file_no_ext = os.path.basename(__file__).split('.')[0]
    uvicorn.run(f"{file_no_ext}:app", host="127.0.0.1", port=8000, reload=True)
