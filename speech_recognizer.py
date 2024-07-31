import speech_recognition as sr

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        #recognizer.adjust_for_ambient_noise(source)
        print("started listening")
        audio = recognizer.listen(source)
        print("finished listening")
        
    try:
        print("started recognizing")
        transcript = recognizer.recognize_google(audio, language="de-DE")
        print("finished recognizing")
        return transcript
    except sr.RequestError:
        return "API unavailable"
    except sr.UnknownValueError:
        return "Unable to recognize speech"
