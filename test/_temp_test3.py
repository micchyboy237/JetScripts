import speech_recognition as sr


def recognize_speech():
    # Create a speech recognition object
    r = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Please say something:")
        # Listen for audio from the microphone
        audio = r.listen(source)

        try:
            # Use Google's speech recognition API to transcribe the audio
            transcription = r.recognize_google(audio)
            print("You said: " + transcription)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand your audio")
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition")


def continuous_recognition():
    # Create a speech recognition object
    r = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Please say something (say 'stop' to exit):")
        while True:
            # Listen for audio from the microphone
            audio = r.listen(source)

            try:
                # Use Google's speech recognition API to transcribe the audio
                transcription = r.recognize_google(audio)
                print("You said: " + transcription)

                if transcription.lower() == "stop":
                    break
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand your audio")
            except sr.RequestError:
                print("Could not request results from Google Speech Recognition")


if __name__ == "__main__":
    # Call the recognize_speech function for a single transcription
    # recognize_speech()

    # Call the continuous_recognition function for continuous transcription
    continuous_recognition()
