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
# Compute similarity
# ----------------------------
logger.info("Calculating similarity between speakers...")
similarity = tool.similarity(speaker_file1, speaker_file2)
logger.info(f"Cosine similarity: {similarity:.6f}")
