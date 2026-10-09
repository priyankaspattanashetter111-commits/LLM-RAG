
import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
import tempfile
import os

from pathlib import Path
from typing import List
from dotenv import load_dotenv

from elevenlabs.client import ElevenLabs

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"
WHISPER_MODEL = "tiny"

CHROMA_DIRECTORY = "knowledge_base"
COLLECTION_NAME = "voice_rag_assistant"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

RETRIEVAL_K = 4

SAMPLE_RATE = 44100
DEFAULT_RECORDING_DURATION = 5


# ============================================================
# DOCUMENT PROCESSOR
# ============================================================

class DocumentProcessor:

    def __init__(self):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

        # Local free embeddings using Ollama
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL
        )


    def load_documents(self, directory: str) -> List[Document]:
        """
        Load documents from different file types.
        """

        documents = []

        for file_path in Path(directory).rglob("*"):

            if not file_path.is_file():
                continue

            extension = file_path.suffix.lower()

            try:

                if extension == ".pdf":

                    loader = PyPDFLoader(str(file_path))
                    loaded_docs = loader.load()

                elif extension == ".txt":

                    loader = TextLoader(
                        str(file_path),
                        encoding="utf-8"
                    )

                    loaded_docs = loader.load()

                elif extension == ".csv":

                    loader = CSVLoader(str(file_path))
                    loaded_docs = loader.load()

                elif extension == ".docx":

                    loader = Docx2txtLoader(str(file_path))
                    loaded_docs = loader.load()

                else:

                    continue

                documents.extend(loaded_docs)

                print(
                    f"Loaded {file_path.name}"
                )

            except Exception as e:

                print(
                    f"Error loading {file_path.name}: {e}"
                )

        return documents


    def process_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:

        """
        Split documents into smaller chunks.
        """

        return self.text_splitter.split_documents(
            documents
        )


    def create_vector_store(
        self,
        documents: List[Document],
        persist_directory: str
    ) -> Chroma:

        """
        Create and persist ChromaDB vector store.
        """

        os.makedirs(
            persist_directory,
            exist_ok=True
        )

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=COLLECTION_NAME
        )

        return vector_store


# ============================================================
# VOICE GENERATOR - ELEVENLABS
# ============================================================

class VoiceGenerator:

    def __init__(self, api_key):

        self.client = ElevenLabs(
            api_key=api_key
        )

        # Available ElevenLabs voices
        self.available_voices = [
            "Rachel",
            "Domi",
            "Bella",
            "Antoni",
            "Elli",
            "Josh",
            "Arnold",
            "Adam",
            "Sam"
        ]

        self.default_voice = "Rachel"


    def generate_voice_response(
        self,
        text: str,
        voice_name: str = None
    ) -> str:

        """
        Generate speech using ElevenLabs.
        """

        try:

            selected_voice = (
                voice_name
                or self.default_voice
            )

            # ElevenLabs voice IDs
            voice_ids = {

                "Rachel":
                    "21m00Tcm4TlvDq8ikWAM",

                "Domi":
                    "AZnzlk1XvdvUeBnXmlld",

                "Bella":
                    "EXAVITQu4vr4xnSDxMaL",

                "Antoni":
                    "ErXwobaYiN019PkySvjV",

                "Elli":
                    "MF3mGyEYCl7XYWbV9V6O",

                "Josh":
                    "TxGEqnHWrfWFTfGW9XjX",

                "Arnold":
                    "VR6AewLTigWG4xSOukaG",

                "Adam":
                    "pNInz6obpgDQGcFmaJgB",

                "Sam":
                    "yoZ06aMxZJJ28mfd3POQ"
            }

            voice_id = voice_ids.get(
                selected_voice,
                voice_ids["Rachel"]
            )

            audio_generator = (
                self.client
                .text_to_speech
                .convert(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128"
                )
            )

            audio_bytes = b"".join(
                audio_generator
            )

            with tempfile.NamedTemporaryFile(
                suffix=".mp3",
                delete=False
            ) as temp_audio:

                temp_audio.write(
                    audio_bytes
                )

                return temp_audio.name

        except Exception as e:

            print(
                f"Error generating voice response: {e}"
            )

            return None


# ============================================================
# VOICE ASSISTANT RAG
# ============================================================

class VoiceAssistantRAG:

    def __init__(self, elevenlabs_api_key):

        # Local Whisper
        self.whisper_model = whisper.load_model(
            WHISPER_MODEL
        )

        # Local Ollama LLM
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0
        )

        # Local Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL
        )

        self.vector_store = None

        self.sample_rate = SAMPLE_RATE

        # ElevenLabs
        self.voice_generator = VoiceGenerator(
            elevenlabs_api_key
        )


    def setup_vector_store(
        self,
        vector_store
    ):

        """
        Initialize the vector store.
        """

        self.vector_store = vector_store


    def load_existing_vector_store(self):

        """
        Load an existing ChromaDB database.
        """

        if not os.path.exists(
            CHROMA_DIRECTORY
        ):

            return False

        try:

            self.vector_store = Chroma(
                persist_directory=CHROMA_DIRECTORY,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )

            return True

        except Exception as e:

            print(
                f"Error loading ChromaDB: {e}"
            )

            return False


    def record_audio(
        self,
        duration=DEFAULT_RECORDING_DURATION
    ):

        """
        Record audio from microphone.
        """

        recording = sd.rec(
            int(
                duration * self.sample_rate
            ),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        return recording


    def transcribe_audio(
        self,
        audio_array
    ):

        """
        Transcribe audio using local Whisper.
        """

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp_audio:

                temp_path = temp_audio.name

            sf.write(
                temp_path,
                audio_array,
                self.sample_rate
            )

            result = self.whisper_model.transcribe(
                temp_path,
                fp16=False
            )

            return result["text"].strip()

        finally:

            if temp_path and os.path.exists(
                temp_path
            ):

                os.unlink(temp_path)


    def generate_response(
        self,
        query
    ):

        """
        Generate RAG response using
        ChromaDB + Ollama.
        """

        if self.vector_store is None:

            return (
                "Error: Vector store "
                "not initialized."
            )

        # Retrieve relevant documents
        documents = self.vector_store.similarity_search(
            query,
            k=RETRIEVAL_K
        )

        if not documents:

            return (
                "I could not find relevant "
                "information in the uploaded documents."
            )

        # Build context
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are a helpful voice assistant.

Answer the user's question using ONLY
the information from the provided context.

Rules:
1. Do not invent information.
2. If the answer is not present in the context,
   say that the information is not available
   in the uploaded documents.
3. Give a clear and concise answer.
4. Make the response natural for voice output.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content.strip()


    def text_to_speech(
        self,
        text: str,
        voice_name: str = None
    ) -> str:

        """
        Convert text to speech using ElevenLabs.
        """

        return (
            self.voice_generator
            .generate_voice_response(
                text,
                voice_name
            )
        )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

def setup_knowledge_base():

    st.title(
        "📚 Knowledge Base Setup"
    )

    doc_processor = DocumentProcessor()

    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=[
            "pdf",
            "txt",
            "csv",
            "docx"
        ]
    )

    if uploaded_files and st.button(
        "Process Documents",
        type="primary"
    ):

        with st.spinner(
            "Processing documents..."
        ):

            temp_dir = tempfile.mkdtemp()

            # --------------------------------
            # Save uploaded files
            # --------------------------------

            for file in uploaded_files:

                file_path = os.path.join(
                    temp_dir,
                    file.name
                )

                with open(
                    file_path,
                    "wb"
                ) as f:

                    f.write(
                        file.getbuffer()
                    )

            try:

                # --------------------------------
                # Load documents
                # --------------------------------

                documents = (
                    doc_processor
                    .load_documents(
                        temp_dir
                    )
                )

                if not documents:

                    st.error(
                        "No supported documents were found."
                    )

                    return

                # --------------------------------
                # Split documents
                # --------------------------------

                processed_docs = (
                    doc_processor
                    .process_documents(
                        documents
                    )
                )

                # --------------------------------
                # Create vector store
                # --------------------------------

                vector_store = (
                    doc_processor
                    .create_vector_store(
                        processed_docs,
                        CHROMA_DIRECTORY
                    )
                )

                # --------------------------------
                # Store in session state
                # --------------------------------

                st.session_state.vector_store = (
                    vector_store
                )

                st.success(
                    f"✅ Processed "
                    f"{len(processed_docs)} "
                    f"document chunks!"
                )

            except Exception as e:

                st.error(
                    f"Error processing documents: {e}"
                )

            finally:

                # --------------------------------
                # Cleanup temporary files
                # --------------------------------

                for file in os.listdir(
                    temp_dir
                ):

                    file_path = os.path.join(
                        temp_dir,
                        file
                    )

                    if os.path.isfile(
                        file_path
                    ):

                        os.remove(
                            file_path
                        )

                os.rmdir(
                    temp_dir
                )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.set_page_config(
        page_title="Voice RAG Assistant",
        page_icon="🎙️",
        layout="wide"
    )

    # --------------------------------
    # Check ElevenLabs API key
    # --------------------------------

    elevenlabs_api_key = os.getenv(
        "ELEVEN_LABS_API_KEY"
    )

    if not elevenlabs_api_key:

        st.error(
            "Please set ELEVEN_LABS_API_KEY "
            "in your .env file."
        )

        st.info(
            "Example:\n\n"
            "ELEVEN_LABS_API_KEY=your_api_key"
        )

        return

    # --------------------------------
    # Navigation
    # --------------------------------

    st.sidebar.title(
        "🧭 Navigation"
    )

    page = st.sidebar.radio(
        "Go to",
        [
            "Setup Knowledge Base",
            "Voice Assistant"
        ]
    )

    # ========================================================
    # KNOWLEDGE BASE PAGE
    # ========================================================

    if page == "Setup Knowledge Base":

        setup_knowledge_base()

    # ========================================================
    # VOICE ASSISTANT PAGE
    # ========================================================

    else:

        if "vector_store" not in st.session_state:

            # Try loading existing ChromaDB
            assistant_loader = (
                VoiceAssistantRAG(
                    elevenlabs_api_key
                )
            )

            loaded = (
                assistant_loader
                .load_existing_vector_store()
            )

            if loaded:

                st.session_state.vector_store = (
                    assistant_loader.vector_store
                )

            else:

                st.error(
                    "Please setup the knowledge base first!"
                )

                return

        st.title(
            "🎙️ Voice Assistant RAG System"
        )

        st.write(
            "Ask questions about your uploaded "
            "documents using your voice."
        )

        # --------------------------------
        # Initialize assistant
        # --------------------------------

        if "assistant" not in st.session_state:

            with st.spinner(
                "Loading local AI models..."
            ):

                st.session_state.assistant = (
                    VoiceAssistantRAG(
                        elevenlabs_api_key
                    )
                )

                st.session_state.assistant.setup_vector_store(
                    st.session_state.vector_store
                )

        assistant = (
            st.session_state.assistant
        )

        # --------------------------------
        # Voice selection
        # --------------------------------

        st.sidebar.subheader(
            "🔊 Voice Settings"
        )

        available_voices = (
            assistant
            .voice_generator
            .available_voices
        )

        selected_voice = st.sidebar.selectbox(
            "Select Voice",
            available_voices,
            index=(
                available_voices.index("Rachel")
                if "Rachel" in available_voices
                else 0
            )
        )

        # --------------------------------
        # Recording duration
        # --------------------------------

        duration = st.slider(
            "Recording Duration",
            min_value=3,
            max_value=15,
            value=DEFAULT_RECORDING_DURATION,
            step=1
        )

        # --------------------------------
        # Recording buttons
        # --------------------------------

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🎤 Start Recording",
                use_container_width=True
            ):

                with st.spinner(
                    f"Recording for {duration} seconds..."
                ):

                    audio_data = (
                        assistant.record_audio(
                            duration
                        )
                    )

                    st.session_state.audio_data = (
                        audio_data
                    )

                    st.success(
                        "✅ Recording completed!"
                    )

        with col2:

            if st.button(
                "▶️ Process Recording",
                use_container_width=True
            ):

                if "audio_data" not in st.session_state:

                    st.error(
                        "Please record audio first!"
                    )

                    return

                # --------------------------------
                # Transcription
                # --------------------------------

                with st.spinner(
                    "📝 Transcribing..."
                ):

                    query = (
                        assistant
                        .transcribe_audio(
                            st.session_state.audio_data
                        )
                    )

                    st.write(
                        "**You said:**",
                        query
                    )

                if not query:

                    st.warning(
                        "Could not understand the audio."
                    )

                    return

                # --------------------------------
                # RAG response
                # --------------------------------

                with st.spinner(
                    "🧠 Generating response with Llama..."
                ):

                    try:

                        response = (
                            assistant
                            .generate_response(
                                query
                            )
                        )

                        st.write(
                            "**Response:**",
                            response
                        )

                        st.session_state.last_response = (
                            response
                        )

                        # --------------------------------
                        # Chat history
                        # --------------------------------

                        if "chat_history" not in st.session_state:

                            st.session_state.chat_history = []

                        st.session_state.chat_history.append(
                            (
                                query,
                                response
                            )
                        )

                    except Exception as e:

                        st.error(
                            f"Error generating response: {e}"
                        )

                        return

                # --------------------------------
                # ElevenLabs TTS
                # --------------------------------

                with st.spinner(
                    "🔊 Converting response to speech..."
                ):

                    audio_file = (
                        assistant
                        .text_to_speech(
                            response,
                            selected_voice
                        )
                    )

                if audio_file:

                    st.subheader(
                        "🔊 Voice Response"
                    )

                    st.audio(
                        audio_file,
                        format="audio/mp3"
                    )

                    # Do not immediately delete
                    # before Streamlit finishes
                    # using the file.

                else:

                    st.error(
                        "Failed to generate voice response."
                    )

        # --------------------------------
        # Chat history
        # --------------------------------

        if "chat_history" in st.session_state:

            st.divider()

            st.subheader(
                "💬 Chat History"
            )

            for q, a in reversed(
                st.session_state.chat_history
            ):

                st.markdown(
                    f"**You:** {q}"
                )

                st.markdown(
                    f"**Assistant:** {a}"
                )

                st.divider()

            if st.button(
                "🗑️ Clear Chat History"
            ):

                st.session_state.chat_history = []

                st.rerun()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

