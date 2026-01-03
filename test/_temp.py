import librosa
import numpy as np
from scipy.spatial import distance
def compare_audio_similarity(file1, file2):
    # Load audio files
    signal1, sr1 = librosa.load(file1)
    signal2, sr2 = librosa.load(file2)
    # Ensure both signals have the same sample rate
    if sr1 != sr2:
        signal2 = librosa.resample(signal2, sr2, sr1)
    # Calculate MFCCs
    mfccs1 = librosa.feature.mfcc(signal1, sr=sr1)
    mfccs2 = librosa.feature.mfcc(signal2, sr=sr1)
    # Calculate the mean of MFCCs
    mean_mfccs1 = np.mean(mfccs1, axis=1)
    mean_mfccs2 = np.mean(mfccs2, axis=1)
    # Calculate the cosine similarity
    similarity = 1 - distance.cosine(mean_mfccs1, mean_mfccs2)
    return similarity
# Example usage
file1 = 'audio_file1.wav'
file2 = 'audio_file2.wav'
similarity = compare_audio_similarity(file1, file2)
print(f'Audio similarity: {similarity:.4f}')