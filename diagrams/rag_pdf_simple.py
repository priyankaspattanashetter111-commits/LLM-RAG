import hashlib
import uuid
from pathlib import Path

import PyPDF2
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = BASE_DIR / "chroma_db"


# ============================================================
# MODEL SELECTOR
# ============================================================

class SimpleModelSelector:

    def __init__(self):

        # ----------------------------------------------------
        # Ollama LLM models
        # ----------------------------------------------------

        self.llm_models = {

            "llama3.2:latest": "Llama 3.2",

            "gemma3:4b": "Gemma 3 4B",

            "qwen3:4b": "Qwen 3 4B",

        }

        # ----------------------------------------------------
        # Embedding models
        # ----------------------------------------------------

        self.embedding_models = {

            "all-minilm": {
                "name": "All-MiniLM",
                "model_name": "all-minilm",
            },

            "nomic": {
                "name": "Nomic Embed Text v2 MoE",
                "model_name": "nomic-embed-text-v2-moe:latest",
            },

            "chroma": {
                "name": "Chroma Default",
                "model_name": "Chroma Default",
            },

        }

    def select_models(self):

        st.sidebar.title("🤖 Model Selection")

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        llm = st.sidebar.selectbox(

            "Choose LLM Model:",

            options=list(
                self.llm_models.keys()
            ),

            format_func=lambda x:
                self.llm_models[x],

        )

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        embedding = st.sidebar.selectbox(

            "Choose Embedding Model:",

            options=list(
                self.embedding_models.keys()
            ),

            format_func=lambda x:
                self.embedding_models[x]["name"],

        )

        return llm, embedding


# ============================================================
# PDF PROCESSOR
# ============================================================

class SimplePDFProcessor:
    """
    Reads PDF files and creates text chunks.
    """

    def __init__(
        self,
        chunk_size=1200,
        chunk_overlap=100
    ):

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

    # ========================================================
    # READ PDF
    # ========================================================

    def read_pdf(self, pdf_file):

        pages = []

        reader = PyPDF2.PdfReader(
            pdf_file
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text()

            if page_text:

                pages.append(
                    (
                        page_number,
                        page_text.strip()
                    )
                )

        return pages

    # ========================================================
    # CREATE CHUNKS
    # ========================================================

    def create_chunks(
        self,
        pages,
        pdf_file
    ):

        chunks = []

        for page_number, text in pages:

            if not text:
                continue

            start = 0

            text_length = len(text)

            while start < text_length:

                end = min(
                    start + self.chunk_size,
                    text_length
                )

                chunk = text[
                    start:end
                ]

                # ------------------------------------------------
                # Try to end at a sentence
                # ------------------------------------------------

                if end < text_length:

                    last_period = chunk.rfind(".")

                    if last_period > 300:

                        chunk = (
                            chunk[
                                :last_period + 1
                            ]
                        )

                        end = (
                            start
                            + last_period
                            + 1
                        )

                chunk = chunk.strip()

                if chunk:

                    chunks.append(

                        {
                            "id": str(
                                uuid.uuid4()
                            ),

                            "text": chunk,

                            "metadata": {

                                "source":
                                    pdf_file.name,

                                "page":
                                    page_number,

                            },
                        }

                    )

                # ------------------------------------------------
                # Move forward
                # ------------------------------------------------

                next_start = (
                    end
                    - self.chunk_overlap
                )

                if next_start <= start:

                    next_start = end

                start = next_start

        return chunks


# ============================================================
# RAG SYSTEM
# ============================================================

class SimpleRAGSystem:

    def __init__(
        self,
        embedding_model="all-minilm",
        llm_model="llama3.2:latest"
    ):

        self.embedding_model = (
            embedding_model
        )

        self.llm_model = (
            llm_model
        )

        # ----------------------------------------------------
        # ChromaDB
        # ----------------------------------------------------

        self.db = chromadb.PersistentClient(

            path=str(
                CHROMA_PATH
            )

        )

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        self.setup_embedding_function()

        # ----------------------------------------------------
        # Ollama LLM
        # ----------------------------------------------------

        self.llm = OpenAI(

            base_url=(
                "http://localhost:11434/v1"
            ),

            api_key="ollama",

        )

        # ----------------------------------------------------
        # Collection
        # ----------------------------------------------------

        self.collection = (
            self.setup_collection()
        )

    # ========================================================
    # EMBEDDING FUNCTION
    # ========================================================

    def setup_embedding_function(self):

        # ----------------------------------------------------
        # All-MiniLM
        # ----------------------------------------------------

        if self.embedding_model == "all-minilm":

            self.embedding_fn = (

                embedding_functions
                .OllamaEmbeddingFunction(

                    url=(
                        "http://localhost:11434/"
                        "api/embeddings"
                    ),

                    model_name="all-minilm",

                )

            )

        # ----------------------------------------------------
        # Nomic
        # ----------------------------------------------------

        elif self.embedding_model == "nomic":

            self.embedding_fn = (

                embedding_functions
                .OllamaEmbeddingFunction(

                    url=(
                        "http://localhost:11434/"
                        "api/embeddings"
                    ),

                    model_name=(
                        "nomic-embed-text-v2-moe:latest"
                    ),

                )

            )

        # ----------------------------------------------------
        # Chroma Default
        # ----------------------------------------------------

        elif self.embedding_model == "chroma":

            self.embedding_fn = (

                embedding_functions
                .DefaultEmbeddingFunction()

            )

        else:

            raise ValueError(
                f"Unsupported embedding model: "
                f"{self.embedding_model}"
            )

    # ========================================================
    # COLLECTION
    # ========================================================

    def setup_collection(self):

        collection_name = (
            f"pdf_documents_{self.embedding_model}"
        )

        try:

            collection = (
                self.db.get_collection(
                    name=collection_name,
                    embedding_function=(
                        self.embedding_fn
                    ),
                )
            )

            return collection

        except Exception:

            collection = (
                self.db.create_collection(

                    name=collection_name,

                    embedding_function=(
                        self.embedding_fn
                    ),

                    metadata={
                        "embedding_model":
                            self.embedding_model
                    },

                )
            )

            return collection

    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        chunks
    ):

        if not chunks:

            return False

        try:

            ids = [
                chunk["id"]
                for chunk in chunks
            ]

            documents = [
                chunk["text"]
                for chunk in chunks
            ]

            metadatas = [
                chunk["metadata"]
                for chunk in chunks
            ]

            # ------------------------------------------------
            # Add in batches
            #
            # This prevents very large embedding requests.
            # ------------------------------------------------

            batch_size = 20

            total = len(documents)

            progress = st.progress(
                0
            )

            status = st.empty()

            for start in range(
                0,
                total,
                batch_size
            ):

                end = min(
                    start + batch_size,
                    total
                )

                self.collection.add(

                    ids=ids[
                        start:end
                    ],

                    documents=documents[
                        start:end
                    ],

                    metadatas=metadatas[
                        start:end
                    ],

                )

                progress.progress(
                    end / total
                )

                status.write(
                    f"Embedding chunks "
                    f"{end} / {total}"
                )

            progress.empty()
            status.empty()

            return True

        except Exception as e:

            st.error(
                "Error adding documents: "
                f"{str(e)}"
            )

            return False

    # ========================================================
    # QUERY DOCUMENTS
    # ========================================================

    def query_documents(
        self,
        query,
        n_results=3
    ):

        try:

            document_count = (
                self.collection.count()
            )

            if document_count == 0:

                return None

            n_results = min(
                n_results,
                document_count
            )

            results = (
                self.collection.query(

                    query_texts=[
                        query
                    ],

                    n_results=n_results,

                    include=[
                        "documents",
                        "metadatas",
                        "distances",
                    ],

                )
            )

            return results

        except Exception as e:

            st.error(
                "Error querying documents: "
                f"{str(e)}"
            )

            return None

    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    def generate_response(
        self,
        query,
        context
    ):

        try:

            context_text = (
                "\n\n".join(
                    context
                )
            )

            prompt = f"""
Answer the question using ONLY the context below.

If the answer is not available in the context,
say:

"The uploaded document does not contain this information."

Do not invent information.

CONTEXT:
-------------------------
{context_text}
-------------------------

QUESTION:
{query}

ANSWER:
"""

            response = (

                self.llm
                .chat
                .completions
                .create(

                    model=self.llm_model,

                    messages=[

                        {
                            "role": "system",

                            "content": (
                                "You answer questions "
                                "about uploaded documents. "
                                "Use only the supplied "
                                "context."
                            ),

                        },

                        {
                            "role": "user",

                            "content": prompt,

                        },

                    ],

                    temperature=0.1,

                )

            )

            return (
                response
                .choices[0]
                .message
                .content
            )

        except Exception as e:

            st.error(
                "Error generating response: "
                f"{str(e)}"
            )

            return None

    # ========================================================
    # EMBEDDING INFORMATION
    # ========================================================

    def get_embedding_info(self):

        if self.embedding_model == "all-minilm":

            return {

                "name":
                    "All-MiniLM",

                "model":
                    "all-minilm",

            }

        if self.embedding_model == "nomic":

            return {

                "name":
                    "Nomic Embed Text v2 MoE",

                "model":
                    "nomic-embed-text-v2-moe:latest",

            }

        return {

            "name":
                "Chroma Default",

            "model":
                "Chroma Default",

        }


# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(
    uploaded_file
):

    file_bytes = (
        uploaded_file.getvalue()
    )

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ============================================================
# MAIN STREAMLIT APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # PAGE CONFIG
    # --------------------------------------------------------

    st.set_page_config(

        page_title="PDF RAG System",

        page_icon="📚",

        layout="wide",

    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.title(
        "📚 PDF RAG System"
    )

    st.write(
        "Upload a PDF and ask questions "
        "using your local Ollama models."
    )

    # ========================================================
    # SESSION STATE
    # ========================================================

    if "rag_system" not in st.session_state:

        st.session_state.rag_system = None

    if "current_embedding_model" not in st.session_state:

        st.session_state.current_embedding_model = None

    if "current_llm_model" not in st.session_state:

        st.session_state.current_llm_model = None

    if "processed_file_hash" not in st.session_state:

        st.session_state.processed_file_hash = None

    if "processed_file_name" not in st.session_state:

        st.session_state.processed_file_name = None

    # ========================================================
    # MODEL SELECTION
    # ========================================================

    model_selector = (
        SimpleModelSelector()
    )

    llm_model, embedding_model = (
        model_selector.select_models()
    )

    # ========================================================
    # SIDEBAR INFORMATION
    # ========================================================

    st.sidebar.divider()

    st.sidebar.markdown(
        "### ⚙️ Current Configuration"
    )

    st.sidebar.write(
        f"**LLM:** {llm_model}"
    )

    st.sidebar.write(
        f"**Embedding:** "
        f"{embedding_model}"
    )

    # ========================================================
    # CHECK CONFIGURATION CHANGE
    # ========================================================

    configuration_changed = (

        st.session_state.current_embedding_model
        != embedding_model

        or

        st.session_state.current_llm_model
        != llm_model

    )

    if configuration_changed:

        st.session_state.rag_system = None

        st.session_state.processed_file_hash = None

        st.session_state.processed_file_name = None

        st.session_state.current_embedding_model = (
            embedding_model
        )

        st.session_state.current_llm_model = (
            llm_model
        )

    # ========================================================
    # INITIALIZE RAG
    # ========================================================

    try:

        if st.session_state.rag_system is None:

            with st.spinner(
                "Loading RAG system..."
            ):

                st.session_state.rag_system = (

                    SimpleRAGSystem(

                        embedding_model=(
                            embedding_model
                        ),

                        llm_model=(
                            llm_model
                        ),

                    )

                )

    except Exception as e:

        st.error(
            "Could not initialize RAG system."
        )

        st.exception(e)

        return

    # ========================================================
    # PDF UPLOAD
    # ========================================================

    st.markdown(
        "## 📄 Upload PDF"
    )

    pdf_file = st.file_uploader(

        "Choose a PDF file",

        type=["pdf"],

    )

    # ========================================================
    # PROCESS PDF
    # ========================================================

    if pdf_file:

        current_hash = (
            get_file_hash(
                pdf_file
            )
        )

        # ----------------------------------------------------
        # Process only once
        # ----------------------------------------------------

        if (
            current_hash
            != st.session_state.processed_file_hash
        ):

            processor = (
                SimplePDFProcessor(
                    chunk_size=1200,
                    chunk_overlap=100
                )
            )

            with st.spinner(
                "Reading PDF..."
            ):

                try:

                    # ----------------------------------------
                    # Read pages
                    # ----------------------------------------

                    pages = (
                        processor.read_pdf(
                            pdf_file
                        )
                    )

                    if not pages:

                        st.error(
                            "No readable text was "
                            "found in this PDF."
                        )

                        return

                    st.success(
                        f"Extracted text from "
                        f"{len(pages)} pages."
                    )

                    # ----------------------------------------
                    # Create chunks
                    # ----------------------------------------

                    chunks = (
                        processor.create_chunks(
                            pages,
                            pdf_file
                        )
                    )

                    st.info(
                        f"Created "
                        f"{len(chunks)} chunks."
                    )

                    # ----------------------------------------
                    # Add to ChromaDB
                    # ----------------------------------------

                    with st.spinner(
                        "Creating embeddings..."
                    ):

                        success = (

                            st.session_state
                            .rag_system
                            .add_documents(
                                chunks
                            )

                        )

                    if success:

                        st.session_state.processed_file_hash = (
                            current_hash
                        )

                        st.session_state.processed_file_name = (
                            pdf_file.name
                        )

                        st.success(
                            f"Successfully processed "
                            f"{pdf_file.name}"
                        )

                except Exception as e:

                    st.error(
                        "Error processing PDF: "
                        f"{str(e)}"
                    )

                    st.exception(e)

    # ========================================================
    # DOCUMENT STATUS
    # ========================================================

    if (
        st.session_state.processed_file_name
    ):

        st.divider()

        st.markdown(
            "## 📑 Processed Document"
        )

        st.write(
            "✅ "
            + st.session_state.processed_file_name
        )

        count = (
            st.session_state
            .rag_system
            .collection
            .count()
        )

        st.caption(
            f"Stored chunks: {count}"
        )

    # ========================================================
    # QUESTION SECTION
    # ========================================================

    if (
        st.session_state.processed_file_hash
    ):

        st.divider()

        st.markdown(
            "## 🔍 Query Your Document"
        )

        query = st.text_input(

            "Ask a question:",

            placeholder=(
                "e.g. What are the objectives?"
            ),

        )

        if st.button(
            "Get Answer",
            type="primary"
        ):

            if not query.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                # --------------------------------------------
                # RETRIEVAL
                # --------------------------------------------

                with st.spinner(
                    "Searching document..."
                ):

                    results = (

                        st.session_state
                        .rag_system
                        .query_documents(

                            query,

                            n_results=3

                        )

                    )

                if (

                    results

                    and

                    results.get("documents")

                    and

                    results["documents"][0]

                ):

                    documents = (
                        results["documents"][0]
                    )

                    metadatas = (
                        results.get(
                            "metadatas",
                            [[]]
                        )[0]
                    )

                    distances = (
                        results.get(
                            "distances",
                            [[]]
                        )[0]
                    )

                    # ----------------------------------------
                    # GENERATE ANSWER
                    # ----------------------------------------

                    with st.spinner(
                        "Generating answer..."
                    ):

                        response = (

                            st.session_state
                            .rag_system
                            .generate_response(

                                query,

                                documents

                            )

                        )

                    if response:

                        st.markdown(
                            "### 💬 Answer"
                        )

                        st.write(
                            response
                        )

                    # ----------------------------------------
                    # SOURCES
                    # ----------------------------------------

                    with st.expander(
                        "📖 View Source Passages"
                    ):

                        for idx, doc in enumerate(
                            documents
                        ):

                            st.markdown(
                                f"**Passage "
                                f"{idx + 1}**"
                            )

                            if (
                                idx
                                < len(metadatas)
                            ):

                                metadata = (
                                    metadatas[idx]
                                )

                                source = (
                                    metadata.get(
                                        "source",
                                        "Unknown"
                                    )
                                )

                                page = (
                                    metadata.get(
                                        "page",
                                        "Unknown"
                                    )
                                )

                                st.caption(
                                    f"Source: "
                                    f"{source} | "
                                    f"Page: {page}"
                                )

                            if (
                                idx
                                < len(distances)
                            ):

                                st.caption(
                                    "Distance: "
                                    f"{distances[idx]:.4f}"
                                )

                            st.info(
                                doc
                            )

                else:

                    st.warning(
                        "No relevant passages "
                        "were found."
                    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "ℹ️ Model Information"
    ):

        st.write(
            f"**LLM:** {llm_model}"
        )

        embedding_info = (
            st.session_state
            .rag_system
            .get_embedding_info()
        )

        st.write(
            f"**Embedding:** "
            f"{embedding_info['name']}"
        )

        st.write(
            f"**Embedding Model:** "
            f"{embedding_info['model']}"
        )

        st.write(
            "**LLM Provider:** Ollama"
        )

        st.write(
            "**Vector Database:** ChromaDB"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

