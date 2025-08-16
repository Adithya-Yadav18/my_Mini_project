# app.py
import streamlit as st
import os
import requests
from streamlit_mic_recorder import mic_recorder
from llm_handler import rewrite_text
from tts_handler import text_to_speech
import io
import wave
from streamlit_lottie import st_lottie
import PyPDF2
import docx
from pydub import AudioSegment
from googletrans import Translator

st.set_page_config(
    page_title="EchoVerse",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Lottie Animation Loader ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Custom CSS for Styling ---
def load_css():
    st.markdown("""
        <style>
            
            @keyframes gradientMove {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .stApp { 
                background: linear-gradient(-45deg, #0E1117, #1a1a2e, #16213e, #0E1117); 
                background-size: 400% 400%; 
                animation: gradientMove 15s ease infinite;
                color: #FAFAFA; 
            }
            [data-testid="stSidebar"] { 
                background: linear-gradient(180deg, #1a1a2e, #16213e); 
            }
            [data-testid="stMarkdownContainer"] canvas {
                background: transparent !important;
            }
            [data-testid="stMarkdownContainer"] {
                background: transparent !important;
            }
            .section-card {
                background: rgba(255, 255, 255, 0.08); 
                border-radius: 15px; 
                padding: 25px; 
                border: 1px solid rgba(255, 255, 255, 0.2); 
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
                backdrop-filter: blur(8px);
                animation: fadeInUp 0.6s ease forwards;
            }
            h1.gradient-title {
                background: linear-gradient(90deg, #00c6ff, #0072ff, #ff4b2b, #ff416c);
                background-size: 300% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: gradientMove 8s ease infinite;
                text-align: center;
                font-weight: bold;
                font-size: 2.8em;
            }
            .stButton>button { 
                border-radius: 10px; 
                border: 1px solid #00c6ff; 
                background: linear-gradient(90deg, #0072ff, #00c6ff); 
                color: white; 
                font-weight: bold; 
                transition: all 0.3s ease;
            }
            .stButton>button:hover { 
                background: linear-gradient(90deg, #00c6ff, #0072ff); 
                box-shadow: 0 0 15px #00c6ff; 
                transform: scale(1.05);
            }
        </style>
    """, unsafe_allow_html=True)

def gradient_title(text):
    st.markdown(f"<h1 class='gradient-title'>{text}</h1>", unsafe_allow_html=True)
        
# --- Text Extraction Functions ---
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# --- Transcription and Translation with Whisper API ---
def transcribe_and_translate_with_api(audio_bytes, hf_token):
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "audio/wav"
    }
    translator = Translator()
    
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_in_memory = io.BytesIO()
        audio_segment.export(wav_in_memory, format="wav")
        wav_in_memory.seek(0)
        
        response = requests.post(API_URL, headers=headers, data=wav_in_memory)
        
        if response.status_code == 200:
            result = response.json()
            transcribed_text = result.get("text", "")
            if transcribed_text:
                # Dedicated translation step
                translated = translator.translate(transcribed_text, dest='en')
                return translated.text
            else:
                st.error("The API returned an empty transcription. Please try speaking again.")
                return ""
        else:
            st.error(f"Transcription API Error: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        st.error(f"An error occurred during audio processing: {e}")
        return ""

def main():
    load_css()
    
    gradient_title("EchoVerse 🎙️")
    st.markdown("<p style='text-align: center; font-size: 1.1em; color: #cccccc;'>Your AI-Powered Audiobook Creator</p>", unsafe_allow_html=True)
    
    # --- Lottie Animation moved to main content area ---
    lottie_url = "https://jsonkeeper.com/b/RZECE"
    lottie_json = load_lottieurl(lottie_url)
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    if lottie_json is not None:
        st_lottie(lottie_json, height=250, key="center_animation")
    else:
        st.warning("Lottie animation failed to load.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")

    hf_token = ""
    token_loaded = False
    try:
        hf_token = st.secrets["HUGGINGFACE_TOKEN"]
        if hf_token:
            token_loaded = True
    except (KeyError, FileNotFoundError):
        pass

    with st.sidebar:
        st.header("Controls")
        
        if not token_loaded:
            st.error("Hugging Face token not found!")
            st.info("Please add your token to .streamlit/secrets.toml to continue.")
            st.stop()
        
        st.success("Hugging Face token loaded!")
        st.write("---")
        
        st.subheader("1. Select Tone & Voice")
        tone = st.radio("Choose a Tone", ('Neutral', 'Suspenseful', 'Inspiring'), help="Select the emotional tone.")
        voice = st.selectbox("Choose a Voice", ('Lisa (Female)', 'Michael (Male)', 'Allison (Female)'), help="Select the voice.")
        
        st.write("---")
        st.subheader("2. Generate Audiobook")
        generate_button = st.button("✨ Generate Audiobook", use_container_width=True, type="primary")
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📝 Provide Your Text Content")
    with col2:
        st.subheader("📄 Upload a Document")
    
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""

    col1_content, col2_content = st.columns(2)
    with col1_content:
        with st.container():
            st.markdown("#### 🎤 Record Audio (Any Language)")
            audio_info = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key='recorder')
            if audio_info and audio_info['bytes']:
                audio_bytes = audio_info['bytes']

                with st.spinner("Transcribing and translating audio..."):
                    transcribed_text = transcribe_and_translate_with_api(audio_bytes, hf_token)
                    if transcribed_text:
                        st.session_state.text_input = transcribed_text

    with col2_content:
            with st.container():
                uploaded_file = st.file_uploader(" ", type=["txt", "pdf", "docx"], label_visibility="collapsed")
            if uploaded_file:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                if file_ext == "txt":
                    st.session_state.text_input = uploaded_file.getvalue().decode("utf-8")
                elif file_ext == "pdf":
                    st.session_state.text_input = extract_text_from_pdf(uploaded_file)
                elif file_ext == "docx":
                    st.session_state.text_input = extract_text_from_docx(uploaded_file)

    st.text_area("Or paste your text here:", value=st.session_state.text_input, height=200, key="text_area_input")
    
    st.session_state.text_input = st.session_state.text_area_input
    original_text = st.session_state.text_input

    if generate_button and original_text:
        with st.spinner("Rewriting text... This may take a moment."):
            rewritten_text = rewrite_text(original_text, tone)
        st.success("Text rewriting complete!")

        if rewritten_text and "Failed" not in rewritten_text:
            with st.spinner("Generating audio..."):
                audio_file_path = text_to_speech(rewritten_text, voice)
            st.success("Audio generation complete!")

            with st.expander("🔊 View Results & Listen", expanded=True):
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.info("Original Text")
                    st.write(original_text)
                with res_col2:
                    st.success(f"Rewritten Text ({tone})")
                    st.write(rewritten_text)
                st.write("---")
                if audio_file_path:
                    audio_file = open(audio_file_path, "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
                    st.download_button("Download MP3", audio_bytes, f"echoverse_{tone.lower()}.mp3", "audio/mp3")
    elif generate_button and not original_text:
        st.error("Please provide some text first.")

if __name__ == "__main__":
    main()