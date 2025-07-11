import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import threading
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import io
from pydub import AudioSegment
from pydub.playback import play as pydub_play
import tempfile
import base64

# Import RAG logic
from rag_module import (
    extract_text,
    split_text,
    create_vector_store,
    get_relevant_chunks,
    answer_query_with_context,
)

# Import multimodal logic
from MultimodInput import get_user_query

# Import concept diagram generator logic
from diagramgen import generate_diagram_streamlit


# --- Load environment ---
load_dotenv()
# eleven_api_key = os.getenv("ELEVEN_API_KEY")
eleven_api_key = st.secrets("ELEVEN_API_KEY")

# --- Streamlit UI setup ---
st.set_page_config(page_title="Groq Chatbot", layout="centered")
st.title("🤖 AI-Powered Learning Assistant")
st.caption("Multimodal input: text, voice, or image → Groq LLM response → optional voice output.")

# --- Sidebar ---
with st.sidebar:
    # api_key = st.text_input("🔑 Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    api_key = st.text_input("🔑 Groq API Key", type="password", value=st.secrets("GROQ_API_KEY", ""))
    # eleven_api_key = st.text_input("🗝️ ElevenLabs API Key", type="password", value=os.getenv("ELEVEN_API_KEY", ""))
    eleven_api_key = st.text_input("🗝️ ElevenLabs API Key", type="password", value=st.secrets("ELEVEN_API_KEY", ""))
    model = st.selectbox("🧠 Choose Model", [
        "mistral-saba-24b", 
        "llama3-70b-8192", 
    ], index=0)

    input_mode = st.radio("🎛️ Input Type", ["Text", "Image", "Voice"])
    speak_response = st.checkbox("🔊 Enable AI Voice Output", value=False)

    # Show record button only in Voice mode
    record = False
    if input_mode == "Voice":
        record = st.button("🎙️ Record your voice", key="record_btn")

    generate_diagram_flag = st.sidebar.checkbox("📊 Generate Concept Diagram")

    # st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Document Q&A")
    uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    use_doc_context = st.sidebar.checkbox("🔍 Use document context")
    # custom_summary_prompt = st.sidebar.text_input("🧾 Custom summarization prompt")
    # trigger_summary = st.sidebar.button("Generate")

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Get User Query (from multimodal input file) ---
user_query = get_user_query(input_mode, record_button_pressed=record)

if "show_diagram" not in st.session_state:
    st.session_state.show_diagram = True

if api_key and user_query and generate_diagram_flag:
    with st.spinner("🛠️ Generating concept diagram..."):
        diagram_path = generate_diagram_streamlit(user_query)

        if diagram_path:
            with open(diagram_path, "rb") as img_file:
                img_bytes = img_file.read()
                img_b64 = base64.b64encode(img_bytes).decode()

            st.session_state.show_diagram = True

            if st.session_state.show_diagram:
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background-color: #1e1e1e; padding: 10px; border-radius: 8px; width: fit-content;">
                        <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; border-radius: 6px;"/>
                    </div>
                    """, unsafe_allow_html=True)

                    # ⬇️ Download button in main area
                    st.markdown(f"""
                    <div style="background-color: #2a2a2a; padding: 10px; border-radius: 8px; color: white; margin-top: 10px;">
                        <a href="data:image/png;base64,{img_b64}" download="concept_diagram.png"
                        style="color: #00ffcc; text-decoration: none;">⬇️ Click to download concept diagram</a>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    if st.button("✖", key="hide_diagram_btn"):
                        st.session_state.show_diagram = False
        else:
            st.warning("⚠️ Diagram generation failed.")


# --- Voice playback function ---
def play_voice(audio_bytes):
    sound = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    pydub_play(sound)

# --- Chatbot Logic ---

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.doc_chunks = []
    st.session_state.raw_text = ""

# 🔁 Clear if new doc
if uploaded_file:
    if (
        "processed_file_name" in st.session_state and
        st.session_state.get("processed_file_name") != uploaded_file.name
    ):
        for key in ["vector_store", "doc_chunks", "raw_text", "processed_file_name"]:
            st.session_state.pop(key, None)

if uploaded_file:
    # ✅ Only process if it's a new file
    if (
        "processed_file_name" not in st.session_state or
        st.session_state.processed_file_name != uploaded_file.name
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        with st.spinner("Processing document..."):
            raw_text = extract_text(file_path)
            chunks = split_text(raw_text)
            vector_store = create_vector_store(chunks)

            st.session_state.vector_store = vector_store
            st.session_state.doc_chunks = chunks
            st.session_state.raw_text = raw_text
            st.session_state.processed_file_name = uploaded_file.name  # ✅ Track last processed file

        st.success("✅ Document processed and ready!")


if api_key and user_query and not generate_diagram_flag:
    st.session_state.messages.append({"role": "user", "content": user_query})

    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )


        with st.spinner("🤖 Groq is thinking..."):
        # --- Context-Aware Prompt ---
            if use_doc_context and uploaded_file and st.session_state.vector_store:
                context_chunks = get_relevant_chunks(user_query, st.session_state.vector_store)
                final_prompt = f"{user_query}\n\nUse this context:\n" + "\n".join(context_chunks)
                reply = answer_query_with_context(user_query, context_chunks)
            else:
                final_prompt = user_query
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.7,
                    stream=False
                )
                reply = response.choices[0].message.content

            st.session_state.messages.append({"role": "assistant", "content": reply})


        # --- Voice Output ---
        if speak_response and eleven_api_key:
            try:
                eleven_client = ElevenLabs(api_key=eleven_api_key)
                audio_generator = eleven_client.generate(
                    text=reply,
                    voice="Bill",
                    model="eleven_monolingual_v1"
                )
                audio_bytes = b"".join(audio_generator)
                threading.Thread(target=play_voice, args=(audio_bytes,), daemon=True).start()
            except Exception as e:
                st.error(f"🔈 Voice Output Error: {e}")

    except Exception as e:
        st.error(f"❌ LLM Error: {e}")

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not api_key:
    st.info("Please enter your Groq API key in the sidebar to begin.")
