import speech_recognition as sr


def recognize_speech_from_mic():
    # Initialize recognizer class (for recognizing speech)
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Please say something:")
        # Adjust for ambient noise to improve recognition accuracy
        recognizer.adjust_for_ambient_noise(source)
        # Listen for the first phrase and extract it into audio data
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google's free Web Speech API
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from the service; {e}")


if __name__ == "__main__":
    recognize_speech_from_mic()
