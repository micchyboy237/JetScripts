"""
Example: Compute speaker embeddings and similarity using the developer-friendly API.
"""

import os
import logging
from jet.audio.speech.pyannote.speaker_similarity import SpeakerEmbedding

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Initialize tool
# ----------------------------
logger.info("Initializing SpeakerEmbedding tool...")
tool = SpeakerEmbedding(
    model_id="pyannote/embedding",
    token=os.getenv("HF_TOKEN")
)
logger.info("Tool ready.")

# ----------------------------
# Define audio files
# ----------------------------
speaker_file1 = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/pyannote/generated/diarize_file/segments/segment_0003/segment.wav"
speaker_file2 = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/pyannote/generated/diarize_file/segments/segment_0004/segment.wav"

# ----------------------------
# Compute embeddings (whole file)
# ----------------------------
logger.info("Computing embeddings for both speakers...")
embedding1 = tool.embed(speaker_file1)
embedding2 = tool.embed(speaker_file2)

# ----------------------------
# Compute cosine distance
# ----------------------------
logger.info("Calculating cosine distance between speakers...")
distance = tool.distance(speaker_file1, speaker_file2)
logger.info(f"Cosine distance: {distance:.6f}")

# ----------------------------
# Compute similarity
# ----------------------------
logger.info("Calculating similarity between speakers...")
similarity = tool.similarity(speaker_file1, speaker_file2)
logger.info(f"Cosine similarity: {similarity:.6f}")

# ----------------------------
# Example: Compute embedding for a segment
# ----------------------------
segment_embedding = tool.embed(speaker_file1)
logger.info(f"Segment embedding shape: {segment_embedding['vector'].shape}")
