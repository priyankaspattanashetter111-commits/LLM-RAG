import re
import shutil
from pathlib import Path

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG System with Query Expansion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chromadb"
PDF_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# QUERY EXPANDER - COMBINED INTO THIS FILE
# ============================================================

class QueryExpander:
    """Generate three relevant variations of a user's question."""

    def __init__(self, temperature: float = 0.3):
        self.llm = ChatOllama(
            model="llama3.2:latest",
            temperature=temperature,
        )
        self.query_expansion_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
Given the following question, generate exactly 3 different
versions that capture different aspects and perspectives.

Make the variations semantically diverse but relevant.

Original Question:
{question}

Return exactly 3 variations in this format:
1. First variation
2. Second variation
3. Third variation

Do not add explanations.
""",
        )

    def expand_query(self, question: str) -> list[str]:
        question = question.strip()
        if not question:
            return []

        response = self.llm.invoke(
            self.query_expansion_prompt.format(question=question)
        )
        response_text = str(response.content).strip()

        variations = []
        for line in response_text.splitlines():
            match = re.match(r"^\s*\d+[\.\)]\s*(.+?)\s*$", line)
            if match:
                query = match.group(1).strip()
                if query and query.casefold() != question.casefold():
                    if query not in variations:
                        variations.append(query)

        if not variations:
            # Keep the app usable if the model formats its response oddly.
            variations = [question]

        return variations[:3] + [question]


# ============================================================
# SESSION STATE
# ============================================================

if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_queries" not in st.session_state:
    st.session_state.last_queries = []


# ============================================================
# HELPERS
# ============================================================

def initialize_system():
    """Connect to Ollama embeddings and open/create the Chroma collection."""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        vectorstore = Chroma(
            collection_name="rag_documents",
            embedding_function=embeddings,
            persist_directory=str(DB_DIR),
        )
        st.session_state.vectorstore = vectorstore
        st.session_state.initialized = True
        return True, "System initialized successfully."
    except Exception as exc:
        st.session_state.vectorstore = None
        st.session_state.initialized = False
        return False, (
            f"{exc}\n\nCheck that Ollama is running and that the required "
            "models are installed."
        )


def save_uploaded_files(uploaded_files):
    """Save uploaded PDF files into the app's data directory."""
    saved = []
    for uploaded_file in uploaded_files or []:
        if not uploaded_file.name.lower().endswith(".pdf"):
            continue

        # Use only the filename to prevent writing outside PDF_DIR.
        safe_name = Path(uploaded_file.name).name
        destination = PDF_DIR / safe_name
        destination.write_bytes(uploaded_file.getbuffer())
        saved.append(safe_name)
    return saved


def process_documents():
    """Load PDFs, split their text, and add chunks to Chroma."""
    if not st.session_state.initialized or st.session_state.vectorstore is None:
        return False, "Please initialize the system first."

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        return False, f"No PDF files found in: {PDF_DIR}"

    try:
        all_documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_file.name
            all_documents.extend(docs)

        if not all_documents:
            return False, "No text could be extracted from the PDFs."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(all_documents)
        if not chunks:
            return False, "No text chunks were created from the PDFs."

        # Avoid duplicate chunks when the same PDFs are processed repeatedly.
        existing_ids = set()
        try:
            existing = st.session_state.vectorstore.get(include=["metadatas"])
            for metadata in existing.get("metadatas") or []:
                if metadata:
                    existing_ids.add(
                        (metadata.get("source"), metadata.get("page"),
                         metadata.get("chunk_text"))
                    )
        except Exception:
            existing_ids = set()

        new_chunks = []
        for chunk in chunks:
            key = (
                chunk.metadata.get("source"),
                chunk.metadata.get("page"),
                chunk.page_content,
            )
            # Store content in metadata so repeated processing can be detected.
            chunk.metadata["chunk_text"] = chunk.page_content
            if key not in existing_ids:
                new_chunks.append(chunk)

        if new_chunks:
            st.session_state.vectorstore.add_documents(new_chunks)

        st.session_state.documents_processed = True
        return True, (
            f"Processed {len(pdf_files)} PDF(s), created {len(chunks)} chunks, "
            f"and added {len(new_chunks)} new chunks to ChromaDB."
        )
    except Exception as exc:
        return False, (
            f"{exc}\n\nIf this mentions a missing Ollama model, run the "
            "model-pull commands shown below the app."
        )


def reset_database():
    """Close the active reference and delete the local Chroma database."""
    try:
        st.session_state.vectorstore = None
        st.session_state.initialized = False
        st.session_state.documents_processed = False
        st.session_state.last_answer = None
        st.session_state.last_sources = []
        st.session_state.last_queries = []

        if DB_DIR.exists():
            shutil.rmtree(DB_DIR)
        DB_DIR.mkdir(parents=True, exist_ok=True)
        return True, "Database reset successfully. Initialize the system again."
    except Exception as exc:
        return False, str(exc)


def answer_question(question: str):
    """Expand the question, retrieve PDF chunks, and answer from context."""
    if not st.session_state.initialized or st.session_state.vectorstore is None:
        return None, [], [], "System is not initialized. Click Initialize System."

    if not question.strip():
        return None, [], [], "Please enter a question."

    try:
        expander = QueryExpander()
        expanded_queries = expander.expand_query(question)

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        retrieved_documents = []
        seen_content = set()

        for query in expanded_queries:
            for doc in retriever.invoke(query):
                content = doc.page_content.strip()
                if content and content not in seen_content:
                    seen_content.add(content)
                    retrieved_documents.append(doc)

        retrieved_documents = retrieved_documents[:8]
        context = "\n\n".join(doc.page_content for doc in retrieved_documents)

        if not context:
            return (
                "I couldn't find relevant text in the indexed PDFs. "
                "Check that your PDFs contain selectable text and that "
                "you have clicked Process Documents.",
                [],
                expanded_queries,
                None,
            )

        llm = ChatOllama(
            model="llama3.2:latest",
            temperature=0.2,
        )
        prompt = PromptTemplate.from_template(
            """You are a helpful question-answering assistant.
Answer using ONLY the provided context. If the answer is not present,
say that the information is not available in the uploaded documents.
Be clear and concise.

Context:
{context}

Question:
{question}

Answer:"""
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return answer, retrieved_documents, expanded_queries, None
    except Exception as exc:
        return None, [], [], (
            f"{exc}\n\nCheck that Ollama is running and the models "
            "llama3.2:latest and nomic-embed-text:latest are installed."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("System Information")

    st.markdown("**PDF Directory:**")
    st.info(str(PDF_DIR))

    st.markdown("**Database Directory:**")
    st.info(str(DB_DIR))

    st.markdown("**LLM:** `llama3.2:latest`")
    st.markdown("**Embedding model:** `nomic-embed-text:latest`")

    st.divider()
    st.header("Controls")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Save Uploaded PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one PDF first.")
        else:
            try:
                saved = save_uploaded_files(uploaded_files)
                if saved:
                    st.success("Saved: " + ", ".join(saved))
                else:
                    st.warning("No PDF files were saved.")
            except Exception as exc:
                st.error(f"Could not save PDFs: {exc}")

    if st.button("Initialize System", use_container_width=True):
        with st.spinner("Connecting to Ollama and ChromaDB..."):
            ok, message = initialize_system()
        if ok:
            st.success(message)
        else:
            st.error(message)

    if st.button("Process Documents", use_container_width=True):
        with st.spinner("Reading PDFs and adding text chunks to ChromaDB..."):
            ok, message = process_documents()
        if ok:
            st.success(message)
        else:
            st.error(message)

    if st.button("Reset Database", use_container_width=True):
        ok, message = reset_database()
        if ok:
            st.success(message)
        else:
            st.error(message)

    st.divider()
    st.caption("Upload PDFs → Initialize System → Process Documents → Ask a question")


# ============================================================
# MAIN PAGE
# ============================================================

st.title("RAG System with Query Expansion")
st.write(
    "Ask questions about your PDFs. The app expands each question, "
    "retrieves relevant document chunks from ChromaDB, and generates "
    "an answer using your local Ollama model."
)

st.header("System Status")
if st.session_state.initialized:
    st.success("System initialized and connected to ChromaDB.")
else:
    st.warning("System not initialized. Use Initialize System in the sidebar.")

st.header("Query Interface")
with st.form("question_form"):
    question = st.text_area(
        "Enter your question:",
        placeholder="Example: What are the main causes of global warming?",
        height=100,
    )
    submitted = st.form_submit_button("Ask Question", type="primary")

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Expanding query, retrieving documents, and generating answer..."):
            answer, documents, expanded_queries, error = answer_question(question.strip())

        if error:
            st.error(error)
        else:
            st.session_state.last_answer = answer
            st.session_state.last_sources = documents
            st.session_state.last_queries = expanded_queries

if st.session_state.last_answer is not None:
    st.header("Answer")
    st.write(st.session_state.last_answer)

    with st.expander("View Expanded Queries", expanded=True):
        for index, query in enumerate(st.session_state.last_queries, start=1):
            label = "Original question" if index == len(st.session_state.last_queries) else f"Expanded query {index}"
            st.markdown(f"**{label}:** {query}")

    with st.expander("View Retrieved Sources"):
        if st.session_state.last_sources:
            for index, doc in enumerate(st.session_state.last_sources, start=1):
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page", "Unknown")
                st.markdown(f"**Source {index}: {source} — page {page}**")
                st.write(doc.page_content)
                st.divider()
        else:
            st.write("No sources were retrieved.")

st.divider()
st.caption("Powered by Streamlit, LangChain, ChromaDB, and Ollama.")
