async def main():
    from jet.transformers.formatters import format_json
    from IPython.display import HTML, display
    from IPython.display import Image, display
    from PIL import Image
    from agents import Agent, Runner
    from agents import ItemHelpers, MessageOutputItem, trace
    from agents.tool import function_tool
    from datetime import datetime
    from jet.logger import CustomLogger
    from ollama import Ollama
    from pathlib import Path
    from pydantic import BaseModel
    from pymongo.operations import SearchIndexModel
    from typing import Dict
    from typing import Dict, List
    from typing import List
    from typing import List, Optional
    from typing import Optional
    import asyncio
    import base64
    import cv2
    import os
    import pandas as pd
    import pymongo
    import shutil
    import threading
    import time
    import voyageai
    import yt_dlp
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # 🎬 AI-Powered Real-time Video Stream Intelligence & Incident Detection System
    
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/video_intelligence_agent.ipynb)
    
    [![AI Learning Hub For Developers](https://img.shields.io/badge/AI%20Learning%20Hub%20For%20Developers-Click%20Here-blue)](https://www.mongodb.com/resources/use-cases/artificial-intelligence?utm_campaign=ai_learning_hub&utm_source=github&utm_medium=referral)
    
    An intelligent video monitoring system that provides real-time analysis and incident detection for live video streams, broadcasts, and recordings. The system combines  AI embeddings, and languge models to automatically detect, analyze, and resolve video quality issues, network problems, and streaming incidents as they occur.
    
    This notebook implements an **AI-Powered Real-time Video Stream Intelligence & Incident Detection System** that automatically monitors and analyzes video content to detect incidents, quality issues, and network problems. The system extracts frames from videos at regular intervals, generates semantic embeddings using Voyage AI's multimodal models, and creates detailed scene descriptions using Ollama's GPT-4 Vision. 
    
    - **Users can search through video content using natural language queries like "find frames with a referee" and instantly jump to relevant timestamps through an interactive HTML5 video player that displays similarity scores and scene descriptions.**
    
    The system supports multiple video sources including local files, webcams, and YouTube livestreams, storing all analysis data in MongoDB with vector search capabilities for fast retrieval. Built using an agent-based architecture, specialized AI agents handle different aspects like frame retrieval, video display, incident analysis, and stream monitoring. 
    - **Real-time monitoring continuously processes live video feeds, compares frames against a database of known incidents, and provides immediate alerts with visual overlays.**
    
    This makes the system valuable for broadcast monitoring, security surveillance, quality assurance, and content discovery applications where organizations need to automatically detect issues, search large video archives, or monitor live streams for technical problems or security incidents.
    
    Pre-requisite:
    - Please ensure that you are using a MongoDB 8.1 database to use the new $rankFusion operator
    """
    logger.info("# 🎬 AI-Powered Real-time Video Stream Intelligence & Incident Detection System")
    
    # ! pip install -Uq pymongo voyageai pandas datasets opencv-python pillow ollama ollama-agents yt-dlp
    
    # import getpass
    
    
    def set_env_securely(var_name, prompt):
    #     value = getpass.getpass(prompt)
        os.environ[var_name] = value
    
    """
    ## Step 1: Extracting Embeddings and Metadata from Video Data
    
    This step involves extracting video frames and generating corresponding embeddings and metadata descriptions to facilitate intelligent search functionality.
    
    This step covers three key techniques:
    
    - Video-to-frame conversion for image extraction
    - Multimodal embedding generation with Voyage AI to encode semantic relationships between text and images
    - Automated metadata generation using GPT-4o Vision Pro"
    
    ### 1.1 Video to Images Function Explanation
    
    This function: `video_to_images` extracts still images from a video at regular time intervals (default: every 2 seconds).
    
    What it does:
    
    1. Opens a video file using OpenCV
    2. Calculates which frames to extract based on the video's frame rate and desired time interval
    3. Loops through the video, saving only the frames that match the timing interval
    4. Saves each extracted frame as a JPEG with a timestamp filename (e.g., "frame_0001_t2.0s.jpg")
    5. Returns the total number of frames extracted
    
    Key parameters:
    
    - `video_path`: Input video file
    - `output_dir`: Where to save the images (default: "frames")  
    - `interval_seconds`: Time between extractions (default: 2 seconds)
    
    Usage:
    
    The example usage extracts frames every 2 seconds from "videos/video.mp4" and saves them to a "frames" folder. This is useful for creating video thumbnails or analyzing video content frame by frame.
    """
    logger.info("## Step 1: Extracting Embeddings and Metadata from Video Data")
    
    
    
    
    def video_to_images(video_path, output_dir="frames", interval_seconds=2):
        """
        Convert a video to images by extracting frames every specified interval.
    
        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save extracted frames (default: "frames")
            interval_seconds (float): Time interval between frame extractions (default: 2 seconds)
    
        Returns:
            int: Number of frames extracted
        """
    
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
        cap = cv2.VideoCapture(video_path)
    
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
    
        logger.debug(f"Video info: {fps:.2f} FPS, {total_frames} frames, {duration:.2f} seconds")
    
        frame_interval = int(fps * interval_seconds)
    
        frame_count = 0
        extracted_count = 0
    
        try:
            while True:
                ret, frame = cap.read()
    
                if not ret:
                    break
    
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    filename = f"frame_{extracted_count:04d}_t{timestamp:.1f}s.jpg"
                    filepath = os.path.join(output_dir, filename)
    
                    cv2.imwrite(filepath, frame)
                    extracted_count += 1
                    logger.debug(f"Extracted frame {extracted_count}: {filename}")
    
                frame_count += 1
    
        finally:
            cap.release()
    
        logger.debug(f"Extraction complete! {extracted_count} frames saved to '{output_dir}'")
        return extracted_count
    
    """
    In the next cells we will be downloading a video to use for the video intelligence use case
    
    Video for this use case is obtained from YouTube, but you can modify the cells below for your own use case
    """
    logger.info("In the next cells we will be downloading a video to use for the video intelligence use case")
    
    
    
    
    def download_youtube_video(
        url: str, output_dir: str = "videos", quality: str = "best"
    ) -> dict:
        """
        Download a YouTube video to `output_dir` at the given `quality`.
        Falls back to 'best' if the requested format isn't available.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
        def make_opts(fmt):
            return {
                "format": fmt,
                "outtmpl": str(Path(output_dir) / "%(title)s.%(ext)s"),
                "noplaylist": True,
                **{
                    k: False
                    for k in (
                        "extractaudio",
                        "writeinfojson",
                        "writesubtitles",
                        "writeautomaticsub",
                        "ignoreerrors",
                    )
                },
            }
    
        fmt = "best[height<=1080]" if quality == "best" else quality
    
        try:
            with yt_dlp.YoutubeDL(make_opts(fmt)) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    return {"success": False, "error": "No metadata extracted", "url": url}
                if info.get("is_live"):
                    return {
                        "success": False,
                        "error": "Live streams not supported",
                        "title": info.get("title"),
                        "url": url,
                        "is_live": True,
                    }
                if info.get("duration", 0) > 3600:
                    logger.debug("⚠️ Video >1h; this may take a while.")
    
                logger.debug(f"⬇️ Downloading at format `{fmt}`…")
                ydl.download([url])
    
        except yt_dlp.utils.DownloadError as e:
            err = str(e)
            if "Requested format is not available" in err:
                logger.debug(f"⚠️ Format `{fmt}` not found; falling back to `best`.")
                try:
                    with yt_dlp.YoutubeDL(make_opts("best")) as ydl:
                        ydl.download([url])
                        info = ydl.extract_info(url, download=False)
                except Exception as e2:
                    return {
                        "success": False,
                        "error": f"Fallback also failed: {type(e2).__name__}: {e2}",
                        "url": url,
                    }
            else:
                suggestions = {
                    "Video unavailable": "May be private, deleted or geo-blocked",
                    "Sign in to confirm your age": "Requires age verification",
                    "This live event has ended": "Livestream is over",
                    "No video formats found": "No downloadable formats",
                }
                return {
                    "success": False,
                    "error": err,
                    "suggestion": next(
                        (s for k, s in suggestions.items() if k in err),
                        "Check URL and availability",
                    ),
                    "url": url,
                }
    
        files = list(Path(output_dir).glob(f"{info['title']}*.*"))
        if not files:
            return {"success": False, "error": "File not found after download", "url": url}
    
        path = str(files[0])
        size_mb = round(files[0].stat().st_size / (1024**2), 2)
    
        return {
            "success": True,
            "title": info.get("title"),
            "file_path": path,
            "file_size_mb": size_mb,
            "duration": info.get("duration"),
            "uploader": info.get("uploader"),
            "view_count": info.get("view_count"),
            "url": url,
        }
    
    if not os.path.exists("videos"):
        os.makedirs("videos")
    
    video_url = (
        "https://www.youtube.com/watch?v=20DThpeng84"  # This is a video of a football match
    )
    
    download_youtube_video(video_url)
    
    video_title = "video"
    
    video_to_images(
        video_path=f"videos/{video_title}.mp4", output_dir="frames", interval_seconds=2
    )
    
    """
    ### 1.2 Setting environment variables
    """
    logger.info("### 1.2 Setting environment variables")
    
    set_env_securely("VOYAGE_API_KEY", "Enter your Voyage API Key: ")
    
    # set_env_securely("OPENAI_API_KEY", "Enter your Ollama API Key: ")
    
    
    voyageai_client = voyageai.Client()
    
    ollama_client = Ollama()
    
    """
    ### 1.3 Generating Embeddings with Voyage AI
    
    Voyage AI multimodal-3 is a state-of-the-art embedding model that revolutionizes how we process documents containing interleaved text and images by vectorizing both modalities through a unified transformer backbone, eliminating the need for complex document parsing while improving retrieval accuracy by an average of 19.63% over competing models. 
    
    **Unlike traditional CLIP-based models that process text and images separately, voyage-multimodal-3 captures the contextual relationships between visual and textual elements in screenshots, PDFs, slides, tables, and figures, making it ideal for RAG applications and semantic search across content-rich documents where visual layout and textual content are equally important.**
    
    Learn more on Voyage AI Multimodal embeddings here: https://docs.voyageai.com/docs/multimodal-embeddings
    """
    logger.info("### 1.3 Generating Embeddings with Voyage AI")
    
    
    
    
    def get_voyage_embedding(data: Image.Image | str, input_type: str = "document") -> List:
        """
        Get Voyage AI embeddings for images and text.
    
        Args:
            data (Image.Image | str): An image or text to embed.
            input_type (str): Input type, either "document" or "query".
    
        Returns:
            List: Embeddings as a list.
        """
        embedding = voyageai_client.multimodal_embed(
            inputs=[[data]], model="voyage-multimodal-3", input_type=input_type
        ).embeddings[0]
        return embedding
    
    
    
    def encode_image_to_base64(image_path: str) -> str:
        """
        Encode an image file to base64 string for Ollama.
    
        Args:
            image_path (str): Path to the image file
    
        Returns:
            str: Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    
    
    def get_image_embedding(
        image_path: str, input_type: str = "document"
    ) -> Optional[List[float]]:
        """
        Get embedding for a single image file using Voyage AI.
    
        Args:
            image_path (str): Path to the image file
            input_type (str): Input type for embedding ("document" or "query")
    
        Returns:
            Optional[List[float]]: Image embedding vector or None if failed
        """
        try:
            image = Image.open(image_path)
    
            embedding = get_voyage_embedding(image, input_type)
    
            logger.debug(
                f"✓ Got embedding for {os.path.basename(image_path)} (dimension: {len(embedding)})"
            )
            return embedding
    
        except Exception as e:
            logger.debug(f"Error getting embedding for {image_path}: {e}")
            return None
    
    """
    #### 1.3.1 Generating metadata with each from Ollama Vision
    """
    logger.info("#### 1.3.1 Generating metadata with each from Ollama Vision")
    
    def generate_frame_description(image_path: str) -> Optional[str]:
        """
        Generate a description of the frame content using Ollama vision model.
    
        Args:
            image_path (str): Path to the image file
    
        Returns:
            Optional[str]: Description of the frame content or None if failed
        """
        try:
            base64_image = encode_image_to_base64(image_path)
    
            response = ollama_client.chat.completions.create(
                model="llama3.2", log_dir=f"{LOG_DIR}/chats",  # Use GPT-4 vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe what you see in this video frame. Include details about objects, people, actions, setting, and any text visible. Be specific and descriptive but concise.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
    
            description = response.choices[0].message.content
            logger.debug(f"✓ Generated description for {os.path.basename(image_path)}")
            return description
    
        except Exception as e:
            logger.debug(f"Error generating description for {image_path}: {e}")
            return None
    
    """
    ### 1.4 Generating frame metadata
    
    This code processes video frames to extract both AI embeddings and text descriptions for intelligent search capabilities.
    
    **`process_single_frame()`** - Handles individual frame processing:
    1. Loads an image using PIL
    2. Generates a vector embedding using Voyage AI (captures visual semantic meaning)
    3. Creates a text description using Ollama's vision model
    4. Returns both as a dictionary or None if processing fails
    
    **`process_frames_to_embeddings_with_descriptions()`** - Batch processes all frames:
    1. **Discovers frames** - Scans the frames directory for image files (.jpg, .png, etc.)
    2. **Processes sequentially** - Calls `process_single_frame()` for each image
    3. **Adds metadata** - Extracts frame number and timestamp from filename 
    4. **Rate limiting** - Includes delays between API calls to avoid hitting service limits
    5. **Progress tracking** - Shows processing status and handles failures gracefully
    
    The result is a dictionary mapping each frame filename to its embedding vector, text description, frame number, and timestamp - enabling both visual and semantic search capabilities.
    """
    logger.info("### 1.4 Generating frame metadata")
    
    
    
    
    def process_single_frame(
        image_path: str, input_type: str = "document"
    ) -> Optional[Dict]:
        """
        Process a single frame to get both embedding and description.
    
        Args:
            image_path (str): Path to the image file
            input_type (str): Input type for embedding ("document" or "query")
    
        Returns:
            Optional[Dict]: Dictionary with 'embedding' and 'frame_description' or None if failed
        """
        try:
            image = Image.open(image_path)
    
            embedding = get_voyage_embedding(image, input_type)
    
            frame_description = generate_frame_description(image_path)
    
            if embedding is not None and frame_description is not None:
                return {"embedding": embedding, "frame_description": frame_description}
            else:
                logger.debug(f"Failed to process {image_path} - missing embedding or description")
                return None
        except Exception as e:
            logger.debug(f"Error processing frame {image_path}: {e}")
            return None
    
    
    
    def process_frames_to_embeddings_with_descriptions(
        frames_dir: str = "frames",
        input_type: str = "document",
        delay_seconds: float = 0.5,
        cut_off_frame: float = 300,
    ) -> Dict[str, Dict]:
        """
        Process all images in frames folder and get both Voyage AI embeddings and Ollama descriptions.
    
        Args:
            frames_dir (str): Directory containing frame images (default: "frames")
            input_type (str): Input type for embeddings ("document" or "query")
            delay_seconds (float): Delay between API calls to avoid rate limits
    
        Returns:
            Dict[str, Dict]: Dictionary mapping image filenames to {'embedding': [...], 'frame_description': '...'}
        """
    
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory '{frames_dir}' not found")
    
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = []
    
        for file in os.listdir(frames_dir):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(frames_dir, file))
    
        if not image_files:
            logger.debug(f"No image files found in '{frames_dir}'")
            return {}
    
        logger.debug(f"Found {len(image_files)} images in '{frames_dir}'")
        logger.debug("Processing frames for embeddings and descriptions...")
    
        frame_data = {}
        failed_count = 0
    
        for i, image_path in enumerate(sorted(image_files), 1):
            logger.debug(f"\nProcessing {i}/{len(image_files)}: {os.path.basename(image_path)}")
    
            result = process_single_frame(image_path, input_type)
    
            if result is not None:
                result["frame_number"] = i
                filename = os.path.basename(image_path)
                try:
                    timestamp_part = [
                        part
                        for part in filename.split("_")
                        if part.startswith("t") and part.endswith("s.jpg")
                    ][0]
                    result["frame_timestamp"] = float(
                        timestamp_part[1:-5]
                    )  # Remove "t" and "s.jpg"
                    logger.debug(f"✅ Timestamp parsed successfully: {result['frame_timestamp']}")
                except (IndexError, ValueError) as e:
                    logger.debug(
                        f"⚠️ Could not parse timestamp from {filename}, using frame number: {e}"
                    )
                    result["frame_timestamp"] = float(i * 2)  # Assume 2-second intervals
                frame_data[os.path.basename(image_path)] = result
                logger.debug(
                    f"✓ Complete - Embedding: {len(result['embedding'])}D, Description: {len(result['frame_description'])} chars"
                )
            else:
                failed_count += 1
    
            if i < len(image_files):  # Don't delay after the last image
                time.sleep(delay_seconds)
    
            if cut_off_frame is not None and i == cut_off_frame:
                break
    
        logger.debug(f"\n🎉 Completed! Successfully processed {len(frame_data)} frames")
        if failed_count > 0:
            logger.debug(f"⚠️ Failed to process {failed_count} frames")
    
        return frame_data
    
    frame_data = process_frames_to_embeddings_with_descriptions(
        frames_dir="frames", input_type="document", delay_seconds=0.5, cut_off_frame=500
    )
    
    
    frame_data_df = pd.DataFrame.from_dict(frame_data, orient="index")
    
    frame_data_df.head()
    
    """
    ## Step 2: Connecting and Saving Data To MongoDB
    """
    logger.info("## Step 2: Connecting and Saving Data To MongoDB")
    
    set_env_securely("MONGODB_URI", "Enter your MongoDB URI: ")
    
    
    
    def get_mongo_client(mongo_uri):
        """Establish and validate connection to the MongoDB."""
    
        client = pymongo.MongoClient(
            mongo_uri, appname="devrel.showcase.agents.video_intelligence.python"
        )
    
        ping_result = client.admin.command("ping")
        if ping_result.get("ok") == 1.0:
            logger.debug("Connection to MongoDB successful")
            return client
        else:
            logger.debug("Connection to MongoDB failed")
        return None
    
    DB_NAME = "video_intelligence"
    db_client = get_mongo_client(os.environ.get("MONGODB_URI"))
    db = db_client[DB_NAME]
    
    """
    #### 2.1 Collection creation
    """
    logger.info("#### 2.1 Collection creation")
    
    FRAME_INTELLIGENCE_METADATA = "video_intelligence"
    VIDEO_LIBRARY = "video_library"
    PREVIOUS_FRAME_INCIDENTS = "previous_frame_incidentS"
    
    EMBEDDING_DIMENSIONS = 1024
    
    def create_collections():
        existing_collections = db.list_collection_names()
        logger.debug(f"Existing collections: {existing_collections}")
    
        if FRAME_INTELLIGENCE_METADATA not in existing_collections:
            db.create_collection(FRAME_INTELLIGENCE_METADATA)
            logger.debug(f"Created collection: {FRAME_INTELLIGENCE_METADATA}")
        else:
            logger.debug(f"Collection {FRAME_INTELLIGENCE_METADATA} already exists")
    
        if VIDEO_LIBRARY not in existing_collections:
            db.create_collection(VIDEO_LIBRARY)
            logger.debug(f"Created collection: {VIDEO_LIBRARY}")
        else:
            logger.debug(f"Collection {VIDEO_LIBRARY} already exists")
    
        if PREVIOUS_FRAME_INCIDENTS not in existing_collections:
            db.create_collection(PREVIOUS_FRAME_INCIDENTS)
            logger.debug(f"Created collection: {PREVIOUS_FRAME_INCIDENTS}")
        else:
            logger.debug(f"Collection {PREVIOUS_FRAME_INCIDENTS} already exists")
    
    create_collections()
    
    """
    #### 2.2 Creating Vector and Search Indexes
    """
    logger.info("#### 2.2 Creating Vector and Search Indexes")
    
    
    
    def create_vector_search_index(
        collection,
        vector_index_name,
        dimensions=1024,
        quantization="scalar",
        embedding_path="embedding",
    ):
        try:
            existing_indexes = collection.list_search_indexes()
            for index in existing_indexes:
                if index["name"] == vector_index_name:
                    logger.debug(f"Vector search index '{vector_index_name}' already exists.")
                    return
        except Exception as e:
            logger.debug(f"Could not list search indexes: {e}")
            return
    
        index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": embedding_path,
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                }
            ]
        }
    
        if quantization == "scalar":
            index_definition["fields"][0]["quantization"] = quantization
    
        if quantization == "binary":
            index_definition["fields"][0]["quantization"] = quantization
    
        search_index_model = SearchIndexModel(
            definition=index_definition,
            name=vector_index_name,
            type="vectorSearch",
        )
    
        try:
            result = collection.create_search_index(model=search_index_model)
            logger.debug(f"New search index named '{result}' is building.")
        except Exception as e:
            logger.debug(f"Error creating vector search index: {e}")
            return
    
        logger.debug(
            f"Polling to check if the index '{result}' is ready. This may take up to a minute."
        )
        predicate = lambda index: index.get("queryable") is True
    
        while True:
            try:
                indices = list(collection.list_search_indexes(result))
                if indices and predicate(indices[0]):
                    break
                time.sleep(5)
            except Exception as e:
                logger.debug(f"Error checking index readiness: {e}")
                time.sleep(5)
    
        logger.debug(f"{result} is ready for querying.")
    
    """
    This code below defines vector‐search indexes on several collections to support efficient similarity queries over high-dimensional embeddings. 
    
    For each target collection—such as frame metadata, past incident records, and video libraries—it creates named indexes using varying quantization strategies. 
    
    Scalar and binary quantization compress embeddings for reduced storage and faster lookups, while full-fidelity indexes preserve maximum precision at the cost of higher resource usage.
    
    **By configuring multiple index variants on the same collection, you can benchmark and choose the optimal trade-off between search accuracy, speed, and storage footprint.**
    **Once built, these indexes enable rapid nearest‐neighbor retrieval of semantically similar items for tasks like incident detection, frame comparison, and content recommendation.**
    """
    logger.info("This code below defines vector‐search indexes on several collections to support efficient similarity queries over high-dimensional embeddings.")
    
    create_vector_search_index(
        db[FRAME_INTELLIGENCE_METADATA], "vector_search_index_scalar", quantization="scalar"
    )
    create_vector_search_index(
        db[FRAME_INTELLIGENCE_METADATA],
        "vector_search_index_full_fidelity",
        quantization="full_fidelity",
    )
    create_vector_search_index(
        db[FRAME_INTELLIGENCE_METADATA],
        "vector_search_index_binary",
        quantization="binary",
    )
    create_vector_search_index(
        db[PREVIOUS_FRAME_INCIDENTS], "incident_vector_index_scalar", quantization="scalar"
    )
    create_vector_search_index(
        db[VIDEO_LIBRARY], "video_vector_index", quantization="full_fidelity"
    )
    
    """
    The code below is a helper wraps MongoDB Atlas Search index creation: 
    - given a collection, an index-definition dict, and a name, 
    - it builds a SearchIndexModel, calls create_search_index, 
    - and returns the result—printing success or catching errors and returning None.
    """
    logger.info("The code below is a helper wraps MongoDB Atlas Search index creation:")
    
    def create_text_search_index(collection, index_definition, index_name):
        """
        Create a search index for a MongoDB Atlas collection.
    
        Args:
        collection: MongoDB collection object
        index_definition: Dictionary defining the index mappings
        index_name: String name for the index
    
        Returns:
        str: Result of the index creation operation
        """
    
        try:
            search_index_model = SearchIndexModel(
                definition=index_definition, name=index_name
            )
    
            result = collection.create_search_index(model=search_index_model)
            logger.debug(f"Search index '{index_name}' created successfully")
            return result
        except Exception as e:
            logger.debug(f"Error creating search index: {e!s}")
            return None
    
    frame_intelligence_index_definition = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "frame_description": {
                    "type": "string",
                },
                "frame_number": {
                    "type": "number",
                },
                "frame_timestamp": {
                    "type": "date",
                },
            },
        }
    }
    
    create_text_search_index(
        db[FRAME_INTELLIGENCE_METADATA],
        frame_intelligence_index_definition,
        "frame_intelligence_index",
    )
    
    """
    ## Step 3: Data Ingestion
    
    The step starts by clearing out any existing documents in the three target collections (`FRAME_INTELLIGENCE_METADATA`, `PREVIOUS_FRAME_INCIDENTS`, and `VIDEO_LIBRARY`) via repeated calls to `delete_many({})`, ensuring you’re working with a clean slate before seeding new data.
    
    Next, it converts your Pandas DataFrame (`frame_data_df`) into a list of Python dictionaries with `to_dict(orient="records")`, then uses `insert_many` on the `frame_intelligence_collection` (aliased from `db[FRAME_INTELLIGENCE_METADATA]`) to bulk-load those records. 
    
    This pattern guarantees that your frame intelligence collection is freshly populated and ready for downstream tasks like vector indexing or semantic search.
    
    **Because there’s no additional transformation pipeline—no ETL steps, schema migrations, or data-wrangling utilities—loading new data is straightforward.**
    **You simply clear, convert, and insert, which keeps the setup simple and minimizes the chance of errors or mismatches between your source DataFrame and the MongoDB collection.**
    """
    logger.info("## Step 3: Data Ingestion")
    
    db[FRAME_INTELLIGENCE_METADATA].delete_many({})
    db[PREVIOUS_FRAME_INCIDENTS].delete_many({})
    db[VIDEO_LIBRARY].delete_many({})
    
    frame_intelligence_documents = frame_data_df.to_dict(orient="records")
    
    frame_intelligence_collection = db[FRAME_INTELLIGENCE_METADATA]
    
    frame_intelligence_collection.insert_many(frame_intelligence_documents)
    
    """
    ## Step 4: Retrieval Methods
    
    ### 4.1 Semantic Search powered by Vector Search
    
    In the code below **`semantic_search_with_mongodb`** wraps the end-to-end process of running a semantic vector search in MongoDB Atlas. It first obtains a numeric embedding for the user’s query via `get_voyage_embedding`, then constructs a two-stage aggregation pipeline:
    
    1. A **`$vectorSearch`** stage that leverages your precreated vector index to find semantically similar documents.
    2. A **`$project`** stage that strips out the raw embedding and internal `_id`, and injects the similarity score (`vectorSearchScore`) into each result.
       Finally, it executes the pipeline and returns the top-N results as a Python list, abstracting away all of the boilerplate needed to perform high-precision, retrieval-grounded queries.
    
    Under the hood, the MongoDB **`$vectorSearch`** operator supports several key parameters for tuning accuracy and performance:
    
    * **`index`** (string): the name of the vector index to use. ([mongodb.com][1])
    * **`queryVector`** (array): the embedding representing the query text. ([mongodb.com][1])
    * **`path`** (string): the document field that stores precomputed embeddings. ([mongodb.com][1])
    * **`numCandidates`** (int): how many nearest-neighbor candidates to retrieve before final scoring—higher values improve recall at the cost of latency. ([mongodb.com][2])
    * **`limit`** (int): the maximum number of top-scoring documents to return. ([mongodb.com][1])
    
    By tuning `numCandidates` and `limit`, you can balance throughput, resource usage, and retrieval fidelity for your specific dataset.
    
    [1]: https://www.mongodb.com/docs/drivers/rust/v3.1/fundamentals/aggregation/vector-search/ "Atlas Vector Search - Rust Driver v3.1 - MongoDB Docs"
    [2]: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/ "Run Vector Search Queries - Atlas - MongoDB Docs"
    """
    logger.info("## Step 4: Retrieval Methods")
    
    def semantic_search_with_mongodb(
        user_query, collection, top_n=5, vector_search_index_name="vector_search_index"
    ):
        """
        Perform a vector search in the MongoDB collection based on the user query.
    
        Args:
        user_query (str): The user's query string.
        collection (MongoCollection): The MongoDB collection to search.
        top_n (int): The number of top results to return.
        vector_search_index_name (str): The name of the vector search index.
    
        Returns:
        list: A list of matching documents.
        """
    
        query_embedding = get_voyage_embedding(user_query, input_type="query")
    
        if query_embedding is None:
            return "Invalid query or embedding generation failed."
    
        vector_search_stage = {
            "$vectorSearch": {
                "index": vector_search_index_name,  # The vector index we created earlier
                "queryVector": query_embedding,  # The numerical vector representing our query
                "path": "embedding",  # The field containing document embeddings
                "numCandidates": 100,  # Explore this many vectors for potential matches
                "limit": top_n,  # Return only the top N most similar results
            }
        }
    
        project_stage = {
            "$project": {
                "_id": 0,  # Exclude MongoDB's internal ID
                "embedding": 0,
                "score": {
                    "$meta": "vectorSearchScore"  # Include similarity score from vector search
                },
            }
        }
    
        pipeline = [vector_search_stage, project_stage]
    
        results = collection.aggregate(pipeline)
    
        return list(results)
    
    """
    The cells below runs the same semantic search query:
    “Can you get me the frame with the referee on the screen”
    against three different vector‐search index configurations (`scalar`, `full_fidelity`, and `binary`) on the `FRAME_INTELLIGENCE_METADATA` collection. 
    
    Each call to `semantic_search_with_mongodb` embeds the user query, invokes the specified index via MongoDB’s `$vectorSearch`, and returns the top 5 most similar frame documents for that quantization strategy.
    
    **By assigning the results to `scalar_results`, `full_fidelity_results`, and `binary_results`, you can directly compare how each index type affects retrieval quality and performance. This makes it easy to benchmark and choose the optimal trade-off between precision, speed, and storage footprint for your frame‐matching application.**
    """
    logger.info("The cells below runs the same semantic search query:")
    
    user_query = "Can you get me the frame with the refree on the screen"
    
    scalar_results = semantic_search_with_mongodb(
        user_query=user_query,
        collection=db[FRAME_INTELLIGENCE_METADATA],
        top_n=5,
        vector_search_index_name="vector_search_index_scalar",
    )
    
    scalar_results
    
    full_fidelity_results = semantic_search_with_mongodb(
        user_query=user_query,
        collection=db[FRAME_INTELLIGENCE_METADATA],
        top_n=5,
        vector_search_index_name="vector_search_index_full_fidelity",
    )
    
    full_fidelity_results
    
    binary_results = semantic_search_with_mongodb(
        user_query=user_query,
        collection=db[FRAME_INTELLIGENCE_METADATA],
        top_n=5,
        vector_search_index_name="vector_search_index_binary",
    )
    
    binary_results
    
    """
    ### 4.2 Hybrid Search (Text + Vector Search)
    
    **`hybrid_search`** combines semantic vector search and traditional text search in MongoDB using the `$rankFusion` operator. It first converts the `user_query` into an embedding via `get_voyage_embedding`, then defines two sub-pipelines—one using `$vectorSearch` on the specified `vector_search_index_name`, the other using Atlas Search’s `$search` on `text_search_index_name`. These pipelines each retrieve up to 20 candidates, which are then merged and re-ranked according to specified weights, producing a unified list of the top-`top_n` results enriched with detailed scoring information.
    
    The `$rankFusion` stage supports key parameters for fine-tuning relevance blending:
    
    * **`pipelines`**: maps names (“vectorPipeline”, “textPipeline”) to aggregation pipelines that source vector and text matches.
    * **`combination.weights`**: assigns relative importance to each pipeline (e.g. `vector_weight=0.7`, `text_weight=0.3`).
    * **`scoreDetails`**: when set to `true`, includes per-pipeline scores in each document’s `scoreDetails` field.
      After fusion, a `$project` stage hides raw embeddings and internal IDs while surfacing score breakdowns, and a final `$limit` ensures only the top-scoring documents are returned. This abstraction lets you call `hybrid_search(query, collection)` to effortlessly leverage both semantic and lexical matching in one go.
    """
    logger.info("### 4.2 Hybrid Search (Text + Vector Search)")
    
    def hybrid_search(
        user_query,
        collection,
        top_n=5,
        vector_search_index_name="vector_search_index_scalar",
        text_search_index_name="text_search_index",
        vector_weight=0.7,
        text_weight=0.3,
    ):
        """
        Perform hybrid search using both vector and text search with MongoDB RankFusion.
    
        Args:
            user_query (str): The user's query or search term.
            collection (Collection): MongoDB collection object.
            top_n (int): Number of results to return.
            vector_search_index_name (str): Name of the vector search index.
            text_search_index_name (str): Name of the text search index.
            vector_weight (float): Weight for vector search results (0.0-1.0).
            text_weight (float): Weight for text search results (0.0-1.0).
    
        Returns:
            List[Dict]: List of search results with scores and details.
        """
    
        query_embedding = get_voyage_embedding(user_query, input_type="query")
    
        rank_fusion_stage = {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vectorPipeline": [
                            {
                                "$vectorSearch": {
                                    "index": vector_search_index_name,
                                    "path": "embedding",
                                    "queryVector": query_embedding,
                                    "numCandidates": 100,
                                    "limit": 20,
                                }
                            }
                        ],
                        "textPipeline": [
                            {
                                "$search": {
                                    "index": text_search_index_name,
                                    "phrase": {
                                        "query": user_query,
                                        "path": "frame_description",
                                    },
                                }
                            },
                            {"$limit": 20},
                        ],
                    }
                },
                "combination": {
                    "weights": {
                        "vectorPipeline": vector_weight,
                        "textPipeline": text_weight,
                    }
                },
                "scoreDetails": True,
            }
        }
    
        project_stage = {
            "$project": {
                "_id": 0,
                "embedding": 0,
                "scoreDetails": {"$meta": "scoreDetails"},
            }
        }
    
        limit_stage = {"$limit": top_n}
    
        pipeline = [rank_fusion_stage, project_stage, limit_stage]
    
        try:
            results = list(collection.aggregate(pipeline))
    
            logger.debug(f"Found {len(results)} results for query: '{user_query}'")
    
            return results
    
        except Exception as e:
            logger.debug(f"Error executing hybrid search: {e}")
            return []
    
    scalar_hybrid_search_results = hybrid_search(
        user_query=user_query,
        collection=db[FRAME_INTELLIGENCE_METADATA],
        top_n=5,
        vector_search_index_name="vector_search_index_scalar",
        text_search_index_name="text_search_index",
        vector_weight=0.5,
        text_weight=0.5,
    )
    
    scalar_hybrid_search_results
    
    """
    ### 4.3 Viewing the video player and returned time stamp
    
    This code creates an interactive video player for Jupyter notebooks that enables intelligent scene navigation based on AI search results. The `create_video_player_with_scenes()` function takes a video file and search results (containing timestamps, descriptions, and similarity scores), then generates an HTML interface with an embedded video player. It automatically handles video encoding by converting smaller files (under 50MB) to base64 for direct embedding, while serving larger videos from their local path with appropriate MIME type detection.
    
    The interface features a standard HTML5 video player with custom scene navigation controls below it. Users can click timestamp buttons to instantly jump to specific scenes, with each button showing the frame number, timestamp, similarity score, and description preview. When selected, the full scene description appears above the player, the video jumps to that timestamp, and playback begins automatically. The system includes keyboard shortcuts (spacebar for play/pause, arrow keys for 10-second navigation) and visual feedback effects, creating a seamless experience for exploring video content based on AI-generated embeddings and making it ideal for video analysis and content search.
    """
    logger.info("### 4.3 Viewing the video player and returned time stamp")
    
    
    
    
    def create_video_player_with_scenes(
        video_path: str,
        search_results: List[Dict],
        user_query: str = "",
        width: int = 800,
        height: int = 450,
    ) -> None:
        """
        Create an interactive video player with scene navigation for Jupyter notebooks.
    
        Args:
            video_path (str): Path to the video file
            search_results (List[Dict]): Search results with timestamps and descriptions
            user_query (str): The original search query that generated these results
            width (int): Video player width in pixels
            height (int): Video player height in pixels
        """
    
        if not os.path.exists(video_path):
            logger.debug(f"❌ Video file not found: {video_path}")
            return
    
        if not search_results:
            logger.debug("❌ No search results provided")
            return
    
        video_base64 = None
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    
        if file_size_mb < 50:  # Only embed videos smaller than 50MB
            with open(video_path, "rb") as video_file:
                video_data = video_file.read()
                video_base64 = base64.b64encode(video_data).decode()
    
        file_ext = os.path.splitext(video_path)[1].lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".ogg": "video/ogg",
            ".avi": "video/mp4",  # Fallback
            ".mov": "video/mp4",  # Fallback
        }
        video_mime = mime_types.get(file_ext, "video/mp4")
    
        sorted_results = sorted(search_results, key=lambda x: x.get("frame_timestamp", 0))
    
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: {width + 50}px;">
            <h3>🎬 Video Scene Navigator</h3>
    
            <!-- Search Query Display -->
            {f'''<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 18px;">🔍</span>
                    <div>
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 5px;">Search Query:</div>
                        <div style="font-size: 16px; font-weight: bold;">"{user_query}"</div>
                        <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">Found {len(sorted_results)} matching scenes</div>
                    </div>
                </div>
            </div>''' if user_query else ''}
    
            <!-- Video Player -->
            <div style="margin-bottom: 20px; border: 2px solid #ddd; border-radius: 8px; overflow: hidden;">
                <video id="videoPlayer" width="{width}" height="{height}" controls style="display: block;">
                    {"<source src='data:" + video_mime + ";base64," + video_base64 + "' type='" + video_mime + "'>" if video_base64 else "<source src='" + video_path + "' type='" + video_mime + "'>" }
                    Your browser does not support the video tag.
                </video>
            </div>
    
            <!-- Current Scene Info -->
            <div id="currentScene" style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4CAF50;">
                <h4 style="margin: 0 0 10px 0; color: #333;">📍 Current Scene</h4>
                <p id="sceneDescription" style="margin: 0; color: #666; font-style: italic;">Click a timestamp below to view scene details</p>
            </div>
    
            <!-- Scene Navigation Buttons -->
            <div style="background: #f9f9f9; padding: 20px; border-radius: 8px;">
                <h4 style="margin: 0 0 15px 0; color: #333;">🎯 Jump to Scenes</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">
    """
    
        for i, result in enumerate(sorted_results):
            timestamp = result.get("frame_timestamp", 0)
            description = result.get("frame_description", "No description")
            score = result.get("score", 0)
            frame_number = result.get("frame_number", 0)
    
            short_desc = description[:60] + "..." if len(description) > 60 else description
    
            html_content += f"""
                    <button onclick="jumpToScene({timestamp}, `{description.replace('`', "'").replace('"', "'")}`, {score}, {frame_number})"
                            style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; background: white; cursor: pointer; text-align: left; transition: all 0.3s;"
                            onmouseover="this.style.background='#e3f2fd'; this.style.borderColor='#2196F3';"
                            onmouseout="this.style.background='white'; this.style.borderColor='#ddd';">
                        <div style="font-weight: bold; color: #1976D2; margin-bottom: 5px;">
                            ⏱️ {timestamp}s (Frame {frame_number}) | Score: {score:.3f}
                        </div>
                        <div style="font-size: 12px; color: #666;">
                            {short_desc}
                        </div>
                    </button>
    """
    
        html_content += """
                </div>
            </div>
    
            <!-- Video Controls Info -->
            <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                <h4 style="margin: 0 0 10px 0; color: #856404;">💡 How to Use</h4>
                <ul style="margin: 0; color: #856404; font-size: 14px;">
                    <li>Click any timestamp button to jump to that scene</li>
                    <li>Use video controls to play, pause, and adjust volume</li>
                    <li>Scene descriptions appear above when you select a timestamp</li>
                </ul>
            </div>
        </div>
    
        <script>
            function jumpToScene(timestamp, description, score, frameNumber) {
                const video = document.getElementById('videoPlayer');
                const sceneDesc = document.getElementById('sceneDescription');
    
                // Jump to timestamp
                video.currentTime = timestamp;
    
                // Update scene description
                sceneDesc.innerHTML = `
                    <div style="margin-bottom: 10px;">
                        <strong>🎬 Frame ${frameNumber} at ${timestamp}s (Score: ${score.toFixed(3)})</strong>
                    </div>
                    <div style="line-height: 1.5;">
                        ${description}
                    </div>
                `;
    
                // Auto-play if paused
                if (video.paused) {
                    video.play().catch(e => console.log('Auto-play prevented by browser'));
                }
    
                // Scroll video into view
                video.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
    
            // Add time update listener to show current time
            document.getElementById('videoPlayer').addEventListener('timeupdate', function() {
                const currentTime = this.currentTime;
                // You could add real-time scene detection here
            });
    
            // Add keyboard shortcuts
            document.addEventListener('keydown', function(e) {
                const video = document.getElementById('videoPlayer');
    
                switch(e.key) {
                    case ' ':  // Spacebar to play/pause
                        e.preventDefault();
                        if (video.paused) {
                            video.play();
                        } else {
                            video.pause();
                        }
                        break;
                    case 'ArrowLeft':  // Left arrow to go back 10s
                        e.preventDefault();
                        video.currentTime = Math.max(0, video.currentTime - 10);
                        break;
                    case 'ArrowRight':  // Right arrow to go forward 10s
                        e.preventDefault();
                        video.currentTime = Math.min(video.duration, video.currentTime + 10);
                        break;
                }
            });
        </script>
        """
    
        display(HTML(html_content))
    
        logger.debug(f"🎬 Video player created with {len(sorted_results)} scenes")
        logger.debug(f"📁 Video: {os.path.basename(video_path)}")
        if file_size_mb >= 50:
            logger.debug("⚠️  Large video file - serving from local path")
    
    video_path = "videos/video.mp4"  # Replace with your actual video path
    
    logger.debug("🎬 Creating interactive video player...")
    create_video_player_with_scenes(
        video_path,
        full_fidelity_results,
        user_query="Get me a scene where a player is injured or on the ground",
    )
    
    """
    ## Step 5: Making Things Agentic
    """
    logger.info("## Step 5: Making Things Agentic")
    
    # ! pip install -Uq ollama-agents
    
    OPENAI_MODEL = "gpt-4o"
    
    """
    ### 5.1 Creating tools for our agents
    """
    logger.info("### 5.1 Creating tools for our agents")
    
    
    
    @function_tool
    def get_frames_from_scene_description(scene_description: str) -> List[str]:
        """
        Get frames from a scene description provided by the user
    
        Args:
            scene_description (str): The scene description to search for
    
        Returns:
            List[str]: A list of frame numbers that are relevant to the scene
        """
    
        logger.debug(f"Getting frames from scene description: {scene_description}")
    
        results = hybrid_search(
            user_query=scene_description,
            collection=db[FRAME_INTELLIGENCE_METADATA],
            top_n=5,
            vector_search_index_name="vector_search_index_scalar",
            text_search_index_name="text_search_index",
            vector_weight=0.5,
            text_weight=0.5,
        )
    
        logger.debug(f"Frames found: {len(results)} for scene description: {scene_description}")
        return results
    
    @function_tool
    def get_frames_from_scene_image(scene_image: str) -> List[str]:
        """
        Get frames from a scene image provided by the user
    
        Args:
            scene_description (str): The scene description to search for
    
        Returns:
            List[str]: A list of frame numbers that are relevant to the scene
        """
    
        results = hybrid_search(
            user_query=scene_image,
            collection=db[FRAME_INTELLIGENCE_METADATA],
            top_n=5,
            vector_search_index_name="vector_search_index_scalar",
            text_search_index_name="text_search_index",
            vector_weight=0.5,
            text_weight=0.5,
        )
        return results
    
    
    
    
    class FrameData(BaseModel):
        frame_description: str
        frame_number: int
        frame_timestamp: float
        score: float
        filename: str = ""
        embedding: List[float] = []
    
    @function_tool
    def show_video_player_with_frames(frames: List[FrameData], user_query: str) -> str:
        """
        Show the video player with the frame results from the frames intelligence metadata collection.
    
        Args:
            frames (List[FrameData]): A list of frame data objects that are relevant to the scene
            user_query (str): The user query that triggered the intial search and is used to create the video player
    
        Returns:
            str: Success message
        """
        frames_dict = [frame.model_dump() for frame in frames]
    
        video_path = "videos/video.mp4"  # Replace with your actual video path
    
        logger.debug(f"🎬 Creating interactive video player... with query: {user_query}")
        create_video_player_with_scenes(video_path, frames_dict, user_query=user_query)
    
        return f"Video player created with {len(frames)} relevant scenes for query: '{user_query}'"
    
    
    frame_from_description_agent = Agent(
        name="Frame From Descripton Retrieval Agent ",
        instructions="You provide detailed information about frames from a video based on a scene description provided by the user. Always cite your sources.",
        handoff_description="A frame retrieval specialist that takes in a scene description that is a string provided by the user and returns a list of frames that are relevant to the scene",
        tools=[get_frames_from_scene_description],
    )
    
    frame_from_scene_image_agent = Agent(
        name="Frame From Scene Image Retrieval Agent",
        instructions="You provide detailed information about frames from a video based on a scene image provided by the user. Always cite your sources.",
        handoff_description="A frame retrieval specialist that takes in a scene image that is a string provided by the user and returns a list of frames that are relevant to the scene",
        tools=[get_frames_from_scene_image],
    )
    
    show_video_player_agent = Agent(
        name="Show Video Player Agent",
        instructions="You display the video player with the frames that are relevant to the scene description",
        handoff_description="A video player specialist that takes in a list of frames that are relevant to the scene and displays them in a video player",
        tools=[show_video_player_with_frames],
    )
    
    orchestrator_agent = Agent(
        name="video_intelligence_orchestrator",
        instructions=(
            "You are a video intelligence assistant. Your job is to help users by retrieving relevant information using your tools.\n\n"
            "IMPORTANT RULES:\n"
            "1. ALWAYS use frame_from_description_agent when a query mentions a scene description\n"
            "2. ALWAYS use frame_from_scene_image_agent when a query mentions a scene image\n"
            "3. If a query requires BOTH frame from description AND frame from scene image, use BOTH tools in sequence\n"
            "4. NEVER attempt to provide video intelligence information without using your tools\n"
            "5. Each tool provides different types of information - use all appropriate tools for complete assistance"
        ),
        tools=[
            frame_from_description_agent.as_tool(
                tool_name="frame_from_description",
                tool_description="Get frames from a scene description",
            ),
            frame_from_scene_image_agent.as_tool(
                tool_name="frame_from_scene_image",
                tool_description="Get frames from a scene image",
            ),
            show_video_player_agent.as_tool(
                tool_name="show_video_player",
                tool_description="Show the video player with the frames that are relevant to the scene",
            ),
        ],
    )
    
    
    
    async def video_intelligence_assistant(user_query):
        """Run the complete virtual primary care assistant workflow"""
        with trace("Orchestrator evaluator"):
            orchestrator_result = await Runner.run(orchestrator_agent, user_query)
            logger.success(format_json(orchestrator_result))
    
            logger.debug("\n--- Orchestrator Processing Steps ---")
            for item in orchestrator_result.new_items:
                if isinstance(item, MessageOutputItem):
                    text = ItemHelpers.text_message_output(item)
                    if text:
                        logger.debug(f"  - Information gathering step: {text}")
    
            logger.debug(
                f"\n\n--- Final Video Intelligence Response ---\n{orchestrator_result.final_output}"
            )
            logger.debug()
    
        return orchestrator_result.final_output
    
    
    # import nest_asyncio
    
    # nest_asyncio.apply()
    
    def run_video_intelligence_assistant(query):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
        result = loop.run_until_complete(video_intelligence_assistant(query))
    
        loop.close()
    
        return result
    
    query = input("What video intelligence can I help you with today? ")
    run_video_intelligence_assistant(query)
    
    """
    ## Step 6: Live Stream Diagonsis
    """
    logger.info("## Step 6: Live Stream Diagonsis")
    
    
    
    def search_similar_incidents(embedding, collection, threshold=0.7):
        """Search for similar incidents in database"""
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "incident_vector_index_scalar",
                        "path": "embedding",
                        "queryVector": embedding,
                        "numCandidates": 100,
                        "limit": 10,
                    }
                },
                {"$addFields": {"similarity_score": {"$meta": "vectorSearchScore"}}},
                {"$match": {"similarity_score": {"$gte": threshold}}},
            ]
    
            results = list(collection.aggregate(pipeline))
            return results
        except Exception as e:
            logger.debug(f"Error searching incidents: {e}")
            return []
    
    current_monitoring = {
        "active": False,
        "cap": None,
        "frame_count": 0,
        "incidents": [],
        "stats": {"frames_processed": 0, "incidents_detected": 0},
    }
    
    
    
    def _monitoring_worker():
        """Background worker for processing frames"""
        global current_monitoring
    
        try:
            collection = db[PREVIOUS_FRAME_INCIDENTS]
        except Exception as e:
            logger.debug(f"Warning: Could not connect to MongoDB: {e}")
            collection = None
    
        while current_monitoring["active"]:
            try:
                if not current_monitoring["cap"]:
                    break
    
                ret, frame = current_monitoring["cap"].read()
                if not ret:
                    break
    
                current_monitoring["frame_count"] += 1
    
                if (
                    current_monitoring["frame_count"]
    #                 % current_monitoring.get("check_interval", 30)
                    == 0
                ):
                    _process_frame_for_incidents(frame, collection)
    
                time.sleep(0.01)  # Small delay
    
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                break
    
    
    def _process_frame_for_incidents(frame, collection):
        """Process a single frame for incident detection"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            embedding = get_voyage_embedding(pil_image, "query")
    
            if collection is not None:
                incidents = search_similar_incidents(embedding, collection)
    
                for incident in incidents:
                    incident_data = {
                        "incident_type": incident.get("incident_type", "Unknown"),
                        "network_issue": incident.get("network_issue", "Unknown"),
                        "resolution_text": incident.get("resolution_text", "No solution"),
                        "similarity_score": incident.get("similarity_score", 0),
                        "frame_number": current_monitoring["frame_count"],
                        "timestamp": datetime.now().isoformat(),
                    }
    
                    current_monitoring["incidents"].append(incident_data)
                    current_monitoring["stats"]["incidents_detected"] += 1
    
            current_monitoring["stats"]["frames_processed"] += 1
    
        except Exception as e:
            logger.debug(f"Frame processing error: {e}")
    
    
    def _add_incident_overlays(frame):
        """Add incident detection overlays to frame"""
        recent_incidents = current_monitoring["incidents"][-3:]  # Show last 3
    
        if not recent_incidents:
            return frame
    
        overlay_y = 30
        for incident in recent_incidents:
            text_lines = [
                f"⚠️ {incident['incident_type']} (Score: {incident['similarity_score']:.2f})",
                f"Issue: {incident['network_issue'][:40]}...",
                f"Solution: {incident['resolution_text'][:40]}...",
            ]
    
            max_width = max(len(line) for line in text_lines) * 8
            rect_height = len(text_lines) * 20 + 10
    
            cv2.rectangle(
                frame, (10, overlay_y), (max_width, overlay_y + rect_height), (0, 0, 0), -1
            )
            cv2.rectangle(
                frame,
                (10, overlay_y),
                (max_width, overlay_y + rect_height),
                (0, 255, 255),
                2,
            )
    
            for i, line in enumerate(text_lines):
                y_pos = overlay_y + 15 + (i * 20)
                cv2.putText(
                    frame,
                    line,
                    (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
    
            overlay_y += rect_height + 10
    
        stats_text = f"Frames: {current_monitoring['stats']['frames_processed']} | Incidents: {current_monitoring['stats']['incidents_detected']}"
        cv2.putText(
            frame,
            stats_text,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    
        return frame
    
    
    def _print_stats():
        """Print current statistics"""
        stats = current_monitoring["stats"]
        logger.debug(
            f"\n📊 Stats: Frames: {stats['frames_processed']} | Incidents: {stats['incidents_detected']}"
        )
        logger.debug(f"Current frame: {current_monitoring['frame_count']}")
    
    def extract_youtube_url(youtube_url: str) -> Optional[str]:
        """Extract direct stream URL from YouTube"""
        try:
            logger.debug(f"🔍 Extracting stream from: {youtube_url}")
    
            ydl_opts = {
                "format": "best[height<=720]/best",  # Fallback to any quality
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
            }
    
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.debug("📥 Downloading video info...")
                info = ydl.extract_info(youtube_url, download=False)
    
                if not info:
                    logger.debug("❌ No video information extracted")
                    return None
    
                logger.debug(f"📺 Video title: {info.get('title', 'Unknown')}")
                logger.debug(f"📺 Uploader: {info.get('uploader', 'Unknown')}")
                logger.debug(f"🔴 Is live: {info.get('is_live', False)}")
                logger.debug(f"⏱️ Duration: {info.get('duration', 'Unknown')}")
    
                formats = info.get("formats", [])
                logger.debug(f"📊 Found {len(formats)} formats")
    
                if not formats:
                    logger.debug("❌ No video formats available")
                    return None
    
                best_format = None
                for fmt in formats:
                    if fmt.get("vcodec") != "none" and fmt.get("url"):
                        if not best_format or (
                            fmt.get("height", 0) > best_format.get("height", 0)
                        ):
                            best_format = fmt
                            logger.debug(
                                f"📊 Found format: {fmt.get('format_id')} - {fmt.get('width')}x{fmt.get('height')}"
                            )
    
                if not best_format:
                    logger.debug("❌ No suitable video format found")
                    return None
    
                stream_url = best_format["url"]
                logger.debug(f"✅ Stream URL extracted: {stream_url[:60]}...")
                return stream_url
    
        except Exception as e:
            logger.debug(f"💥 YouTube extraction error: {type(e).__name__}: {e}")
    
            if "Video unavailable" in str(e):
                logger.debug("💡 Video may be private, deleted, or geo-blocked")
            elif "Sign in to confirm your age" in str(e):
                logger.debug("💡 Video requires age verification")
            elif "This live event has ended" in str(e):
                logger.debug("💡 Livestream has ended")
            else:
                logger.debug("💡 Try a different YouTube video or check if the URL is correct")
    
            return None
    
    @function_tool
    def stop_video_monitoring() -> str:
        """
        Stop current video monitoring.
    
        Returns:
            Status message with final statistics
        """
        global current_monitoring
    
        if not current_monitoring["active"]:
            return "No active monitoring to stop"
    
        current_monitoring["active"] = False
    
        if current_monitoring["cap"]:
            current_monitoring["cap"].release()
    
        final_stats = {
            "total_frames": current_monitoring["frame_count"],
            "frames_processed": current_monitoring["stats"]["frames_processed"],
            "incidents_detected": current_monitoring["stats"]["incidents_detected"],
        }
    
        cv2.destroyAllWindows()
    
        return f"🛑 Monitoring stopped. Final stats: {final_stats}"
    
    @function_tool
    def start_video_monitoring(source: str = "0", check_interval: int = 30) -> str:
        """
        Start real-time video monitoring for incident detection.
    
        Args:
            source: Video source (0 for webcam, file path, or YouTube URL)
            check_interval: Check every N frames
    
        Returns:
            Status message
        """
        global current_monitoring
    
        try:
            if current_monitoring["active"]:
                stop_video_monitoring()
    
            logger.debug(f"🔍 Processing source: {source}")
    
            if source.startswith("http") and "youtube" in source:
                logger.debug("📺 Detected YouTube URL, extracting stream...")
                stream_url = extract_youtube_url(source)
                if not stream_url:
                    return f"❌ Failed to extract YouTube stream URL from: {source}. Video may be private, restricted, or unavailable."
                logger.debug(f"✅ Extracted stream URL: {stream_url[:60]}...")
                video_source = stream_url
            elif source.isdigit():
                video_source = int(source)  # Webcam
                logger.debug(f"📷 Using webcam: {video_source}")
            else:
                video_source = source  # File path
                logger.debug(f"📁 Using file: {video_source}")
    
            logger.debug("🎥 Initializing video capture...")
            cap = cv2.VideoCapture(video_source)
    
            if not cap.isOpened():
                error_msg = f"❌ Could not open video source: {source}"
                if isinstance(video_source, str) and video_source.startswith("http"):
                    error_msg += "\n   - YouTube stream URL may be invalid or expired"
                    error_msg += (
                        "\n   - Try a different video or check if the stream is live"
                    )
                logger.debug(error_msg)
                return error_msg
    
            logger.debug("🧪 Testing frame capture...")
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                cap.release()
                error_msg = f"❌ Could not read frames from video source: {source}"
                if isinstance(video_source, str) and video_source.startswith("http"):
                    error_msg += "\n   - Stream may have ended or be unavailable"
                logger.debug(error_msg)
                return error_msg
    
            logger.debug(f"✅ Successfully captured test frame: {test_frame.shape}")
    
            current_monitoring["active"] = True
            current_monitoring["cap"] = cap
            current_monitoring["frame_count"] = 0
            current_monitoring["incidents"] = []
            current_monitoring["check_interval"] = check_interval
            current_monitoring["source"] = source
    
            monitor_thread = threading.Thread(target=_monitoring_worker, daemon=True)
            monitor_thread.start()
    
            success_msg = f"✅ Started monitoring source: {source} (checking every {check_interval} frames)"
            logger.debug(success_msg)
            return success_msg
    
        except Exception as e:
            error_msg = f"❌ Error starting monitoring: {e}"
            logger.debug(f"💥 Exception details: {type(e).__name__}: {e}")
            return error_msg
    
    
    
    @function_tool
    def get_monitoring_status() -> dict:
        """
        Get current monitoring status and statistics.
    
        Returns:
            Dictionary with monitoring status and stats
        """
        global current_monitoring
    
        return {
            "active": current_monitoring["active"],
            "frame_count": current_monitoring["frame_count"],
            "frames_processed": current_monitoring["stats"]["frames_processed"],
            "incidents_detected": current_monitoring["stats"]["incidents_detected"],
            "recent_incidents": len(current_monitoring["incidents"]),
            "timestamp": datetime.now().isoformat(),
        }
    
    @function_tool
    def get_recent_incidents(max_incidents: int = 5) -> list:
        """
        Get recent incident detections.
    
        Args:
            max_incidents: Maximum number of incidents to return
    
        Returns:
            List of recent incidents
        """
        global current_monitoring
    
        recent = current_monitoring["incidents"][-max_incidents:]
    
        incidents = []
        for incident in recent:
            incidents.append(
                {
                    "incident_type": incident.get("incident_type", "Unknown"),
                    "network_issue": incident.get("network_issue", "Unknown issue"),
                    "resolution_text": incident.get(
                        "resolution_text", "No solution available"
                    ),
                    "similarity_score": incident.get("similarity_score", 0),
                    "frame_number": incident.get("frame_number", 0),
                    "timestamp": incident.get("timestamp", datetime.now().isoformat()),
                }
            )
    
        return incidents
    
    @function_tool
    def show_video_feed(show_overlays: bool = True) -> str:
        """
        Display the video feed with optional incident overlays.
    
        Args:
            show_overlays: Whether to show incident detection overlays
    
        Returns:
            Status message
        """
        global current_monitoring
    
        if not current_monitoring["active"]:
            return "❌ No active video monitoring to display. Start monitoring first with 'Start monitoring [source]'"
    
        if not current_monitoring["cap"]:
            return "❌ Video capture not available. There may be an issue with the video source."
    
        try:
            logger.debug("🎥 Displaying video feed... (Press 'q' to quit, 's' for stats)")
            logger.debug(f"📺 Source: {current_monitoring.get('source', 'Unknown')}")
            logger.debug(f"🔄 Frame count: {current_monitoring['frame_count']}")
    
            display_count = 0
            max_empty_frames = 10
            empty_frame_count = 0
    
            while current_monitoring["active"]:
                ret, frame = current_monitoring["cap"].read()
                if not ret:
                    empty_frame_count += 1
                    logger.debug(f"⚠️ Failed to read frame {empty_frame_count}/{max_empty_frames}")
    
                    if empty_frame_count >= max_empty_frames:
                        logger.debug("❌ Too many failed frame reads, stopping display")
                        break
    
                    time.sleep(0.1)  # Wait a bit before trying again
                    continue
    
                empty_frame_count = 0
                display_count += 1
    
                if show_overlays:
                    frame = _add_incident_overlays(frame)
    
                info_text = f"Display: {display_count} | Source: {current_monitoring.get('source', 'Unknown')[:30]}"
                cv2.putText(
                    frame,
                    info_text,
                    (10, frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
    
                cv2.imshow("Real-time Video Analysis", frame)
    
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.debug("🛑 Display stopped by user")
                    break
                elif key == ord("s"):
                    _print_stats()
    
            cv2.destroyAllWindows()
            return f"✅ Video display closed after showing {display_count} frames"
    
        except Exception as e:
            cv2.destroyAllWindows()
            error_msg = f"❌ Error displaying video: {e}"
            logger.debug(f"💥 Display error: {type(e).__name__}: {e}")
            return error_msg
    
    @function_tool
    def show_latest_frame() -> str:
        """
        Capture and display the current frame in Jupyter notebook.
    
        Returns:
            Status message with frame info
        """
        global current_monitoring
    
        if not current_monitoring["active"] or not current_monitoring["cap"]:
            return "❌ No active video monitoring to capture frame from"
    
        try:
            ret, frame = current_monitoring["cap"].read()
            if not ret:
                return "❌ Could not capture current frame"
    
            frame = _add_incident_overlays(frame)
    
            info_text = f"Frame: {current_monitoring['frame_count']} | Live capture"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
    
    
            os.makedirs("video_frames", exist_ok=True)
            filename = "video_frames/current_frame.jpg"
            cv2.imwrite(filename, frame)
    
            try:
    
                display(Image(filename))
                return f"✅ Current frame displayed (Frame #{current_monitoring['frame_count']})"
            except ImportError:
                return f"✅ Current frame saved to {filename} (Frame #{current_monitoring['frame_count']})"
    
        except Exception as e:
            return f"❌ Error capturing frame: {e}"
    
    
    @function_tool
    def create_video_summary() -> str:
        """
        Create a summary of the current monitoring session.
    
        Returns:
            Summary of monitoring statistics and recent activity
        """
        global current_monitoring
    
        if not current_monitoring.get("active", False):
            return "❌ No active monitoring session"
    
        try:
            summary = f"""
    📊 **Video Monitoring Summary**
    
    🎬 **Source**: {current_monitoring.get('source', 'Unknown')}
    📊 **Status**: {'🟢 Active' if current_monitoring['active'] else '🔴 Stopped'}
    🔢 **Total Frames**: {current_monitoring.get('frame_count', 0)}
    ⚙️ **Processed**: {current_monitoring.get('stats', {}).get('frames_processed', 0)}
    🚨 **Incidents**: {current_monitoring.get('stats', {}).get('incidents_detected', 0)}
    ⏱️ **Check Interval**: Every {current_monitoring.get('check_interval', 'Unknown')} frames
    
    🎯 **Recent Incidents**: {len(current_monitoring.get('incidents', []))} detected
            """
    
            recent_incidents = current_monitoring.get("incidents", [])[-3:]  # Last 3
            if recent_incidents:
                summary += "\n🚨 **Latest Incidents**:\n"
                for i, incident in enumerate(recent_incidents, 1):
                    summary += f"   {i}. {incident.get('incident_type', 'Unknown')} (Score: {incident.get('similarity_score', 0):.2f})\n"
            else:
                summary += "\n✅ **No Recent Incidents Detected**"
    
            return summary
    
        except Exception as e:
            return f"❌ Error creating summary: {e}"
    
    @function_tool
    def get_youtube_stream_info(youtube_url: str) -> dict:
        """
        Get direct stream information from a YouTube URL.
    
        Args:
            youtube_url: YouTube video/livestream URL
    
        Returns:
            Dictionary with stream information and direct URL
        """
        try:
            ydl_opts = {
                "format": "best[height<=720]",
                "quiet": True,
                "no_warnings": True,
                "extractaudio": False,
            }
    
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
    
                if not info:
                    return {"error": "Could not extract video information"}
    
                formats = info.get("formats", [])
                best_format = None
    
                for fmt in formats:
                    if fmt.get("vcodec") != "none" and fmt.get("url"):
                        if not best_format or (
                            fmt.get("height", 0) > best_format.get("height", 0)
                        ):
                            best_format = fmt
    
                if not best_format:
                    return {"error": "No suitable video format found"}
    
                return {
                    "success": True,
                    "title": info.get("title", "Unknown"),
                    "uploader": info.get("uploader", "Unknown"),
                    "is_live": info.get("is_live", False),
                    "duration": info.get("duration"),
                    "direct_url": best_format["url"],
                    "width": best_format.get("width"),
                    "height": best_format.get("height"),
                    "fps": best_format.get("fps"),
                    "original_url": youtube_url,
                }
    
        except Exception as e:
            return {"error": f"Failed to extract stream: {e}"}
    
    @function_tool
    def show_video_feed_jupyter(
        show_overlays: bool = True, save_frames: bool = True
    ) -> str:
        """
        Display video feed in Jupyter notebook (saves frames instead of using cv2.imshow).
    
        Args:
            show_overlays: Whether to show incident detection overlays
            save_frames: Whether to save frames to files
    
        Returns:
            Status message
        """
        global current_monitoring
    
        if not current_monitoring["active"]:
            return "❌ No active video monitoring to display. Start monitoring first."
    
        if not current_monitoring["cap"]:
            return "❌ Video capture not available."
    
        try:
            logger.debug("🎥 Jupyter-friendly video display starting...")
            logger.debug(f"📺 Source: {current_monitoring.get('source', 'Unknown')}")
            logger.debug(f"🔄 Current frame count: {current_monitoring['frame_count']}")
            logger.debug("📁 Frames will be saved to 'video_frames/' directory")
            logger.debug("🛑 Run 'Stop monitoring' to stop")
    
    
            os.makedirs("video_frames", exist_ok=True)
    
            frame_save_count = 0
            max_frames_to_save = 10  # Save every 30th frame for demo
    
            for i in range(max_frames_to_save):
                if not current_monitoring["active"]:
                    break
    
                ret, frame = current_monitoring["cap"].read()
                if not ret:
                    logger.debug(f"⚠️ Could not read frame {i+1}")
                    continue
    
                if show_overlays:
                    frame = _add_incident_overlays(frame)
    
                info_text = f"Frame: {current_monitoring['frame_count']} | {current_monitoring.get('source', 'Unknown')[:30]}"
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
    
                if save_frames:
                    filename = f"video_frames/frame_{frame_save_count:04d}.jpg"
                    cv2.imwrite(filename, frame)
                    frame_save_count += 1
                    logger.debug(f"💾 Saved frame {frame_save_count}: {filename}")
    
    
                time.sleep(0.5)  # Save a frame every 0.5 seconds
    
            return f"✅ Saved {frame_save_count} frames to 'video_frames/' directory. Monitoring continues in background."
    
        except Exception as e:
            error_msg = f"❌ Error in Jupyter display: {e}"
            logger.debug(f"💥 Display error: {type(e).__name__}: {e}")
            return error_msg
    
    
    monitoring_agent = Agent(
        name="Video Monitoring Agent",
        instructions="You start and stop video monitoring for incident detection. You can handle webcam, files, and YouTube streams.",
        handoff_description="Specialist for starting and managing video stream monitoring",
        tools=[start_video_monitoring, stop_video_monitoring, get_monitoring_status],
    )
    
    incident_agent = Agent(
        name="Incident Analysis Agent",
        instructions="You analyze detected incidents and provide detailed reports about network issues and solutions.",
        handoff_description="Specialist for analyzing and reporting video incidents",
        tools=[get_recent_incidents, get_monitoring_status],
    )
    
    display_agent = Agent(
        name="Video Display Agent",
        instructions="You show the live video feed with incident detection overlays.",
        handoff_description="Specialist for displaying video feeds with overlays",
        tools=[
            show_video_feed_jupyter,
            get_monitoring_status,
            show_latest_frame,
            create_video_summary,
        ],
    )
    
    youtube_agent = Agent(
        name="YouTube Stream Agent",
        instructions="You search for YouTube livestreams and help set up monitoring.",
        handoff_description="Specialist for finding and setting up YouTube livestream monitoring",
        tools=[get_youtube_stream_info, start_video_monitoring],
    )
    
    video_orchestrator = Agent(
        name="Real-time Video Orchestrator",
        instructions=(
            "You coordinate real-time video monitoring and incident detection. "
            "Use monitoring_agent to start/stop monitoring, incident_agent for analysis, "
            "display_agent to show video feeds, and youtube_agent for YouTube streams. "
            "Always provide clear status updates and help users through the complete workflow."
        ),
        tools=[
            monitoring_agent.as_tool(
                tool_name="monitoring",
                tool_description="Start, stop, and manage video monitoring",
            ),
            incident_agent.as_tool(
                tool_name="incidents",
                tool_description="Analyze and report detected incidents",
            ),
            display_agent.as_tool(
                tool_name="display",
                tool_description="Show video feed with incident overlays",
            ),
            youtube_agent.as_tool(
                tool_name="youtube",
                tool_description="Search and monitor YouTube livestreams",
            ),
        ],
    )
    
    async def quick_start_youtube(youtube_url):
        """Quick start with YouTube monitoring"""
        logger.debug(f"🚀 Starting YouTube monitoring: {youtube_url}")
    
        test_result = await Runner.run(
                video_orchestrator, f"Test this YouTube URL: {youtube_url}"
            )
        logger.success(format_json(test_result))
        logger.debug("Test result:", test_result.final_output)
    
        if "test PASSED" in test_result.final_output:
            result = await Runner.run(
                    video_orchestrator, f"Start monitoring this YouTube stream: {youtube_url}"
                )
            logger.success(format_json(result))
            logger.debug("Response:", result.final_output)
    
            logger.debug("\n🎥 Showing video feed...")
            display_result = await Runner.run(
                    video_orchestrator, "Show the video feed with incident overlays"
                )
            logger.success(format_json(display_result))
            logger.debug("Display response:", display_result.final_output)
    
            return result
        else:
            logger.debug("❌ URL test failed, not starting monitoring")
            return test_result
    
    async def get_youtube_info_example(youtube_url):
        """Example: Get YouTube stream information"""
        logger.debug(f"🔍 Getting stream info for: {youtube_url}")
    
        result = await Runner.run(
                video_orchestrator, f"Get stream info for this YouTube URL: {youtube_url}"
            )
        logger.success(format_json(result))
        logger.debug("Response:", result.final_output)
    
        return result
    
    async def video_monitoring_assistant(user_query):
        """Main video monitoring assistant function matching your pattern"""
        try:
        except ImportError:
            result = await Runner.run(video_orchestrator, user_query)
            logger.success(format_json(result))
            return result.final_output
    
        with trace("Video monitoring orchestrator"):
            orchestrator_result = await Runner.run(video_orchestrator, user_query)
            logger.success(format_json(orchestrator_result))
    
            logger.debug("\n--- Video Monitoring Processing Steps ---")
            for item in orchestrator_result.new_items:
                if isinstance(item, MessageOutputItem):
                    text = ItemHelpers.text_message_output(item)
                    if text:
                        logger.debug(f"  - Processing step: {text}")
    
            logger.debug(
                f"\n\n--- Final Video Monitoring Response ---\n{orchestrator_result.final_output}"
            )
            logger.debug()
    
        return orchestrator_result.final_output
    
    async def test_video_assistant():
        """Simple test function to demonstrate usage"""
        logger.debug("🧪 Testing Video Monitoring Assistant")
    
        response1 = await video_monitoring_assistant(
                "What's the current monitoring status?"
            )
        logger.success(format_json(response1))
        logger.debug(f"Status response: {response1}")
    
        response2 = await video_monitoring_assistant(
                "Get stream info for this YouTube URL: https://www.youtube.com/watch?v=jcEW98mEqmY"
            )
        logger.success(format_json(response2))
        logger.debug(f"YouTube info response: {response2}")
    
        return response1, response2
    
    result = await get_youtube_info_example("https://www.youtube.com/watch?v=YDfiTGGPYCk")
    logger.success(format_json(result))
    
    responses = await quick_start_youtube("https://www.youtube.com/watch?v=YDfiTGGPYCk")
    logger.success(format_json(responses))
    
    display_result = await Runner.run(
            video_orchestrator, "Show the video feed with overlays"
        )
    logger.success(format_json(display_result))
    logger.debug(display_result.final_output)
    
    frame_result = await Runner.run(video_orchestrator, "Show the latest frame")
    logger.success(format_json(frame_result))
    
    analysis = await Runner.run(video_orchestrator, "Analyze detected incidents")
    logger.success(format_json(analysis))
    logger.debug(analysis.final_output)
    
    summary = await Runner.run(video_orchestrator, "Create a video summary")
    logger.success(format_json(summary))
    logger.debug(summary.final_output)
    
    result = await Runner.run(video_orchestrator, "Show recent incidents")
    logger.success(format_json(result))
    result.final_output
    
    result = await Runner.run(
            video_orchestrator, "What's the current incident detection status?"
        )
    logger.success(format_json(result))
    result.final_output
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())