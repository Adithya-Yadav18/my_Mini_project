# tts_handler.py
import streamlit as st
from gtts import gTTS
import os

def text_to_speech(text, voice_selection="en"):
    """
    Converts a given text into an MP3 audio file using Google Text-to-Speech.

    Args:
        text (str): The text to be converted to speech.
        voice_selection (str): The language/voice to use for the TTS. 
                               For gTTS, this is primarily language-based.
                               The voice options in the UI are for user experience;
                               we can map them to different language accents if needed.

    Returns:
        str: The file path to the generated MP3 audio file, or None if it fails.
    """
    try:
        # The 'lang' parameter determines the voice. 'en' is standard English.
        # We can add more complex logic here later to map 'Lisa', 'Michael' etc.
        # to different language accents like 'en-us', 'en-uk', 'en-au'.
        # For now, we'll use standard 'en'.
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Define a file path for the output audio
        # Saving in the current directory is simple for Streamlit
        audio_file_path = "temp_audio.mp3"
        
        # Save the audio file
        tts.save(audio_file_path)
        
        return audio_file_path

    except Exception as e:
        st.error(f"An error occurred during text-to-speech conversion: {e}")
        print(f"gTTS Error: {e}")
        return None

