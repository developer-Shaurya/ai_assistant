import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
import tempfile
from PIL import Image
import pytesseract

SAMPLE_RATE = 16000
DURATION = 8

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

def get_user_query(input_mode, record_button_pressed=False):
    user_query = ""

    if input_mode == "Text":
        user_query = st.chat_input("Type your message here...")

    elif input_mode == "Image":
        st.markdown("📤 Upload an image:")
        uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uploaded_image:
            with st.spinner("🔍 Extracting text from image..."):
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width =True)
                text = pytesseract.image_to_string(image)
                if text.strip():
                    # st.success("✅ Text extracted:")
                    # st.text_area("📝 Extracted Text", value=text, height=100)
                    user_query = text.strip()
                else:
                    st.warning("⚠️ No readable text found in the image.")

    elif input_mode == "Voice":
        user_query = ""

        # Create a full container below previous interactions
        with st.container():
            # Create a dynamic placeholder to update status messages
            status_placeholder = st.empty()

            if record_button_pressed:
                # 1️⃣ Show "Listening..." immediately
                status_placeholder.info("🎤 Listening...")

                recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
                sd.wait()

                # 2️⃣ Clear "Listening..." as soon as done
                status_placeholder.empty()

                # 3️⃣ Show "Transcribing..." message
                status_placeholder.info("🧠 Transcribing with Whisper...")

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    sf.write(tmpfile.name, recording, SAMPLE_RATE)
                    result = whisper_model.transcribe(tmpfile.name)
                    user_query = result["text"].strip()

                # 4️⃣ Replace with result
                status_placeholder.empty()

                if user_query:
                    st.success("✅ Transcription complete")
                    # st.text_area("📝 Transcribed Text", value=user_query, height=100)
                else:
                    st.warning("⚠️ Could not transcribe any clear speech.")

    return user_query
