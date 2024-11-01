import discord
import speech_recognition as sr
from pydub import AudioSegment
import io
import ffmpeg

def recognize_speech_from_DF(DiscordAudioFile:discord.File):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    DiscordAudioFile.reset()

    # Save the file locally
    myaudio_segment:AudioSegment = AudioSegment.from_file(io.BytesIO(DiscordAudioFile.fp.read()))

    bytes_audio = io.BytesIO()

    myaudio_segment.export(bytes_audio, format="wav")

    if bytes_audio.getbuffer().nbytes == 0:
        print("The audio data is empty.")
        return "The audio data is empty."

    with sr.AudioFile(bytes_audio) as source:
        audio = recognizer.record(source)
    
    if audio.frame_data == b'':
        print("The audio frame data is empty.")
        return "The audio frame data is empty."

    try:
        transcript = recognizer.recognize_google(audio, language="de-DE") # type: ignore
        print("Transcript:", transcript)
        return transcript
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand the audio")
        return "Google Web Speech API could not understand the audio"
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return f"Could not request results from Google Web Speech API; {e}"


def recognize_speech_from_bytes(bytes:bytes):
    # Initialize recognizer
    recognizer = sr.Recognizer()
    # Save the file locally
    kwarg:dict = {
        'sample_width':2,
        'frame_rate':48000,
        'channels':2
    }
    myaudio_segment:AudioSegment = AudioSegment.from_raw(io.BytesIO(bytes),sample_width=2, frame_rate=48000,channels=2)
    
    bytes_audio = io.BytesIO()

    myaudio_segment.export(bytes_audio, format="wav")

    if bytes_audio.getbuffer().nbytes == 0:
        print("The audio data is empty.")
        return "The audio data is empty."

    with sr.AudioFile(bytes_audio) as source:
        audio = recognizer.record(source)
    
    if audio.frame_data == b'':
        print("The audio frame data is empty.")
        return "The audio frame data is empty."

    try:
        transcript = recognizer.recognize_google(audio, language="de-DE") # type: ignore
        print("Transcript:", transcript)
        return transcript
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand the audio")
        return "Google Web Speech API could not understand the audio"
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return f"Could not request results from Google Web Speech API; {e}"
    
def recognize_speech_from_wav(file_path:str):
    # Initialize recognizer
    recognizer = sr.Recognizer()
    # Save the file locally
    myaudio_segment:AudioSegment = AudioSegment.from_wav(file_path)
    myaudio_segment.set_channels(2)
    myaudio_segment.set_sample_width(2)
    myaudio_segment.set_frame_rate(48000)

    bytes_audio = io.BytesIO()

    myaudio_segment.export(bytes_audio, format="wav")

    if bytes_audio.getbuffer().nbytes == 0:
        print("The audio data is empty.")
        return "The audio data is empty."

    with sr.AudioFile(bytes_audio) as source:
        audio = recognizer.record(source)
    
    if audio.frame_data == b'':
        print("The audio frame data is empty.")
        return "The audio frame data is empty."

    try:
        transcript = recognizer.recognize_google(audio, language="de-DE") # type: ignore
        print("Transcript:", transcript)
        return transcript
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand the audio")
        return "Google Web Speech API could not understand the audio"
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return f"Could not request results from Google Web Speech API; {e}"