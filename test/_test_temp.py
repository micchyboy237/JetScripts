import os
import torch
import torchaudio
import numpy as np
from pyannote.audio import Pipeline, Model, Inference
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from app.core.config import HUGGINGFACE_TOKEN
import warnings
from sqlalchemy import select, update
from app.dependencies.database import SessionLocal
from app.models.model import Download_videos
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from torch.cuda.amp import autocast
warnings.filterwarnings("ignore")

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device that you are using is {device}")

# Load diarization pipeline and embedding model (no .to(device))
try:
    print("Initializing diarization pipeline...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN
    ).to(device)
except Exception as e:
    print(f"Error initializing diarization pipeline : {e}")

try:
    print("Initializing embedding model...")
    embedding_model = Inference(
        Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGINGFACE_TOKEN).to(device), window="whole"
    )
    print("Diarization and embedding models initialized")
except Exception as e:
    print(f"error initializing embedding model: {e}")

# Create a folder for saving embeddings if it doesn't exist
EMBEDDING_DIR = "Your_embedding_save_dir"
if not os.path.exists(EMBEDDING_DIR):
    os.makedirs(EMBEDDING_DIR)
    print(f"Created directory for embeddings: {EMBEDDING_DIR}")

def get_file_name_from_path(file_path):
    """Extract the base file name without the extension from the file path."""
    base_name = os.path.basename(file_path) 
    file_name = os.path.splitext(base_name)[0] 
    return file_name

def get_file_path():
    try:
        session = SessionLocal()
        stmt = select(Download_videos.uuid, Download_videos.location).where(Download_videos.speaker_id == None)
        result = session.execute(stmt)
        download_videos = result.fetchall()
        print("Retrieved audio paths from database:")
        for video in download_videos:
            print(f"UUID: {video.uuid}, Location: {video.location}")
        return download_videos
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def updated_db(uuid, speaker_ids):
    try:
        session = SessionLocal()
        speaker_ids_formatted = [f'speaker_{int(sid)}' for sid in speaker_ids]
        speaker_ids_str = ', '.join(speaker_ids_formatted)
        stmt = update(Download_videos).where(Download_videos.uuid == uuid).values(speaker_id=speaker_ids_str)
        session.execute(stmt)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        session.close()

@contextmanager
def no_grad_context():
    with torch.no_grad():
        yield

def extract_speaker_embeddings(audio_files, save_path):
    all_embeddings = []
    speaker_info = []

    for audio_file in audio_files:
        print(f"\nProcessing file: {audio_file}")

        try: 
            with no_grad_context():
                waveform, sample_rate = torchaudio.load(audio_file)
                waveform = waveform.to(device)
                print(f"Loaded waveform with shape: {waveform.shape}, sample_rate: {sample_rate}")
        except Exception as e:
            print(f"Error loading waveform for {audio_file}: {e}")
            continue

        try:
            with torch.no_grad(), autocast():
            # Perform speaker diarization
                print("Performing speaker diarization...")
                diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
            print("Speaker diarization completed.")
        except Exception:
            print("Error in diarization ")
            continue

        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            duration = end_time - start_time
            if duration < 1.0:
                continue

            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            segment = waveform[:, start_idx:end_idx]
            segment = segment.cpu()

            # Extract embedding
            embedding = embedding_model({"waveform": segment, "sample_rate": sample_rate})
            if not isinstance(embedding, np.ndarray):
                embedding = embedding.numpy()

            embedding = embedding / np.linalg.norm(embedding)
            all_embeddings.append(embedding)
            speaker_info.append({
                'audio_file': audio_file,
                'local_speaker_label': speaker_label,
                'embedding': embedding
            })

            print(f"Extracted embedding for speaker {speaker_label} from {start_time:.2f}s to {end_time:.2f}s.")
        try: 
            np.savez_compressed(save_path, embeddings=all_embeddings, speaker_info=speaker_info)
            print(f"Saved embeddings and speaker info to {save_path}")
        except Exception:
            print("Error is embedding ")

        # Free GPU memory after eacg embedding extraction 
        torch.cuda.empty_cache()

    return all_embeddings, speaker_info

def load_embeddings(save_path):
    data = np.load(save_path, allow_pickle=True)
    all_embeddings = data['embeddings']
    speaker_info = data['speaker_info']
    print(f"Loaded embeddings and speaker info from {save_path}")
    return all_embeddings, speaker_info


def compute_similarity_matrix(embeddings, batch_size=1000):
    try:
            
        """Compute cosine similarity in batches to avoid memory overload."""
        num_embeddings = len(embeddings)
        similarity_matrix = []

        for start_idx in range(0, num_embeddings, batch_size):
            end_idx = min(start_idx + batch_size, num_embeddings)
            batch = embeddings[start_idx:end_idx]
            
            # Calculate similarities for the batch
            batch_similarities = []
            for embedding in embeddings:
                similarities = 1 - cosine_distances(batch, [embedding])
                batch_similarities.append(similarities.flatten())

            similarity_matrix.extend(batch_similarities)
        
        # Convert to NumPy array
        similarity_matrix = np.array(similarity_matrix, dtype=np.float32)
        print("Computed cosine similarity matrix in batches.")
        return similarity_matrix
    except Exception as e:
        print(f"Error in compute similarity matrix: {e}")


def cluster_speaker(embeddings):
    embeddings_array = np.vstack(embeddings)
    print(f"Embeddings array shape: {embeddings_array.shape}")

    cosine_sim_matrix = 1 - cosine_distances(embeddings_array)
    print("Computed cosine similarity matrix.")

    distance_matrix = 1 - cosine_sim_matrix
    clustering = DBSCAN(eps=0.4, min_samples=2, metric='precomputed')
    clustering.fit(distance_matrix)
    labels = clustering.labels_

    print(f"Clustering completed. Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    return labels

def assign_global_speaker_ids(labels, speaker_info):
    speaker_mapping = {}
    for idx, info in enumerate(speaker_info):
        key = (info['audio_file'], info['local_speaker_label'])
        cluster_label = labels[idx]
        if cluster_label == -1:
            continue
        speaker_mapping[key] = cluster_label
    return speaker_mapping

def process_speaker_id():
    download_videos = get_file_path()

    if not download_videos:
        print("No valid audio files found for processing")
        return
    
    all_embeddings = []
    all_speaker_info = []

    # Process each video from the database
    for video in download_videos:
        audio_file = video.location
        audio_file_name = get_file_name_from_path(audio_file)  # Extract file name without extension
        uuid = video.uuid

        # Save embeddings in the 'embeddings' folder with the audio file name
        save_path = os.path.join(EMBEDDING_DIR, f"embeddings_{audio_file_name}.npz")

        # If the embeddings file already exists, load it. Otherwise, extract and save embeddings
        if os.path.exists(save_path):
            embeddings, speaker_info = load_embeddings(save_path)
        else:
            embeddings, speaker_info = extract_speaker_embeddings([audio_file], save_path=save_path)

        all_embeddings.extend(embeddings)
        all_speaker_info.extend(speaker_info)

        #Free GPU memory after processing each file 

        torch.cuda.empty_cache()

    # mow that we have all embeddings, perform clustering and assign speaker IDs
    if all_embeddings:
        cosine_sim_matrix = compute_similarity_matrix(all_embeddings)
        labels = cluster_speaker(all_embeddings)
        speaker_mapping = assign_global_speaker_ids(labels, all_speaker_info)

        file_speakers = {}
        for info in all_speaker_info:
            # Check if 'info' is structured correctly as a dictionary
            if 'audio_file' in info and 'local_speaker_label' in info:
                audio_file = info['audio_file']
                key = (audio_file, info['local_speaker_label'])
                global_speaker_id = speaker_mapping.get(key, -1)
                if global_speaker_id == -1:
                    continue
                if audio_file not in file_speakers:
                    file_speakers[audio_file] = set()
                file_speakers[audio_file].add(global_speaker_id)

        # Update the database for each video
        for video in download_videos:
            audio_file = video.location
            uuid = video.uuid
            speakers_in_file = sorted(file_speakers.get(audio_file, []))
            if speakers_in_file:
                updated_db(uuid, speakers_in_file)

    print("Processing completed")

if __name__ == "__main__":
    process_speaker_id()