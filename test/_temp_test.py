import io
import librosa

# Suppose you have raw audio bytes, e.g. read from a file or received over network
audio_bytes = open("/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/data/sound.wav", "rb").read()

# Wrap in a BytesIO
bio = io.BytesIO(audio_bytes)

# Now pass that file-like object to librosa.load
y, sr = librosa.load(bio, sr=None)  # sr=None keeps native sampling rate
print(y.shape, sr)
