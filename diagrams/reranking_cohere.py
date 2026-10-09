import shutil
from pathlib import Path
from typing import List

import streamlit as st

from dotenv import load_dotenv

from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings
)

from langchain_chroma import Chroma

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.prompts import (
    PromptTemplate
)

from langchain_core.output_parsers import (
    StrOutputParser
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Query Expansion",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).parent

PDF_DIR = BASE_DIR / "data"

DB_DIR = BASE_DIR / "chromadb"


PDF_DIR.mkdir(
    exist_ok=True
)

DB_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "initialized" not in st.session_state:
    st.session_state.initialized = False


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False


# ============================================================
# QUERY EXPANDER
# ============================================================

class QueryExpander:

    def __init__(
        self,
        temperature: float = 0.3
    ):

        # ----------------------------------------------------
        # LOCAL OLLAMA MODEL
        # ----------------------------------------------------

        self.llm = ChatOllama(
            model="llama3.2:latest",
            temperature=temperature
        )

        # ----------------------------------------------------
        # QUERY EXPANSION PROMPT
        # ----------------------------------------------------

        self.query_expansion_prompt = PromptTemplate(
            input_variables=[
                "question"
            ],

            template="""
Given the following question, generate 3 different
versions of the question that capture different
aspects and perspectives of the original question.

Make the variations semantically diverse but relevant.

Original Question:
{question}

Generate exactly 3 variations in the following format:

1. [First variation]
2. [Second variation]
3. [Third variation]

Do not add any explanation.
"""
        )


    # --------------------------------------------------------
    # EXPAND QUERY
    # --------------------------------------------------------

    def expand_query(
        self,
        question: str
    ) -> List[str]:

        try:

            response = self.llm.invoke(
                self.query_expansion_prompt.format(
                    question=question
                )
            )

            response_text = (
                response.content.strip()
            )

            variations = []

            for line in response_text.split("\n"):

                line = line.strip()

                if not line:
                    continue

                # Accept:
                # 1. query
                # 2. query
                # 3. query

                if ". " in line:

                    query = line.split(
                        ". ",
                        1
                    )[1].strip()

                    if query:

                        variations.append(
                            query
                        )

            # ------------------------------------------------
            # ADD ORIGINAL QUESTION
            # ------------------------------------------------

            variations.append(
                question
            )

            return variations

        except Exception as e:

            print(
                f"Error in query expansion: {e}"
            )

            return [
                question
            ]


# ============================================================
# INITIALIZE RAG SYSTEM
# ============================================================

def initialize_system():

    try:

        # ----------------------------------------------------
        # EMBEDDING MODEL
        # ----------------------------------------------------

        embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest"
        )

        # ----------------------------------------------------
        # CHROMA VECTOR DATABASE
        # ----------------------------------------------------

        vectorstore = Chroma(
            collection_name="pdf_collection",

            embedding_function=embeddings,

            persist_directory=str(
                DB_DIR
            )
        )

        # ----------------------------------------------------
        # SAVE TO SESSION STATE
        # ----------------------------------------------------

        st.session_state.vectorstore = (
            vectorstore
        )

        st.session_state.initialized = True

        return (
            True,
            "System initialized successfully."
        )

    except Exception as e:

        return (
            False,
            str(e)
        )


# ============================================================
# PROCESS PDF DOCUMENTS
# ============================================================

def process_documents():

    if not st.session_state.initialized:

        return (
            False,
            "Please initialize the system first."
        )

    # --------------------------------------------------------
    # FIND PDF FILES
    # --------------------------------------------------------

    pdf_files = list(
        PDF_DIR.glob("*.pdf")
    )

    if not pdf_files:

        return (
            False,
            "No PDF files found. Please upload a PDF first."
        )

    try:

        # ----------------------------------------------------
        # LOAD PDF DOCUMENTS
        # ----------------------------------------------------

        all_documents = []

        for pdf_file in pdf_files:

            loader = PyPDFLoader(
                str(pdf_file)
            )

            documents = loader.load()

            all_documents.extend(
                documents
            )

        # ----------------------------------------------------
        # CHECK DOCUMENTS
        # ----------------------------------------------------

        if not all_documents:

            return (
                False,
                "No text could be extracted from the PDFs."
            )

        # ----------------------------------------------------
        # TEXT SPLITTER
        # ----------------------------------------------------

        text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
        )

        chunks = (
            text_splitter.split_documents(
                all_documents
            )
        )

        # ----------------------------------------------------
        # ADD TO CHROMADB
        # ----------------------------------------------------

        st.session_state.vectorstore.add_documents(
            chunks
        )

        # ----------------------------------------------------
        # UPDATE STATE
        # ----------------------------------------------------

        st.session_state.documents_processed = True

        return (
            True,
            f"Successfully processed "
            f"{len(pdf_files)} PDF(s) and created "
            f"{len(chunks)} chunks."
        )

    except Exception as e:

        return (
            False,
            str(e)
        )


# ============================================================
# RESET DATABASE
# ============================================================

def reset_database():

    try:

        # ----------------------------------------------------
        # DELETE CHROMA DATABASE
        # ----------------------------------------------------

        if DB_DIR.exists():

            shutil.rmtree(
                DB_DIR
            )

        # ----------------------------------------------------
        # RECREATE DIRECTORY
        # ----------------------------------------------------

        DB_DIR.mkdir(
            exist_ok=True
        )

        # ----------------------------------------------------
        # RESET SESSION STATE
        # ----------------------------------------------------

        st.session_state.vectorstore = None

        st.session_state.initialized = False

        st.session_state.documents_processed = False

        return (
            True,
            "Database reset successfully."
        )

    except Exception as e:

        return (
            False,
            str(e)
        )


# ============================================================
# SAVE UPLOADED FILES
# ============================================================

def save_uploaded_files(
    uploaded_files
):

    saved_files = []

    for uploaded_file in uploaded_files:

        # ----------------------------------------------------
        # ONLY PDF FILES
        # ----------------------------------------------------

        if not uploaded_file.name.lower().endswith(
            ".pdf"
        ):
            continue

        file_path = (
            PDF_DIR / uploaded_file.name
        )

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        saved_files.append(
            uploaded_file.name
        )

    return saved_files


# ============================================================
# GET DOCUMENT COUNT
# ============================================================

def get_document_count():

    try:

        if (
            st.session_state.vectorstore
            is None
        ):

            return 0

        collection = (
            st.session_state
            .vectorstore
            ._collection
        )

        return collection.count()

    except Exception:

        return 0


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(
    question
):

    if not st.session_state.initialized:

        return (
            None,
            [],
            []
        )

    if st.session_state.vectorstore is None:

        return (
            None,
            [],
            []
        )

    try:

        # ====================================================
        # STEP 1
        # QUERY EXPANSION
        # ====================================================

        expander = QueryExpander()

        expanded_queries = (
            expander.expand_query(
                question
            )
        )


        # ====================================================
        # STEP 2
        # CREATE RETRIEVER
        # ====================================================

        retriever = (
            st.session_state
            .vectorstore
            .as_retriever(
                search_type="similarity",

                search_kwargs={
                    "k": 4
                }
            )
        )


        # ====================================================
        # STEP 3
        # SEARCH WITH ALL QUERIES
        # ====================================================

        retrieved_documents = []

        seen_content = set()


        for query in expanded_queries:

            docs = retriever.invoke(
                query
            )

            for doc in docs:

                content = (
                    doc.page_content
                )

                if content not in seen_content:

                    seen_content.add(
                        content
                    )

                    retrieved_documents.append(
                        doc
                    )


        # ----------------------------------------------------
        # KEEP MAXIMUM 8 DOCUMENTS
        # ----------------------------------------------------

        retrieved_documents = (
            retrieved_documents[:8]
        )


        # ====================================================
        # STEP 4
        # BUILD CONTEXT
        # ====================================================

        context = "\n\n".join(

            doc.page_content

            for doc in retrieved_documents

        )


        # ====================================================
        # STEP 5
        # LOCAL LLM
        # ====================================================

        llm = ChatOllama(
            model="llama3.2:latest",
            temperature=0.2
        )


        # ====================================================
        # STEP 6
        # RAG PROMPT
        # ====================================================

        prompt = PromptTemplate.from_template(
            """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the
provided context.

If the answer is not present in the context,
say that the information is not available
in the uploaded documents.

Be clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""
        )


        # ====================================================
        # STEP 7
        # CREATE RAG CHAIN
        # ====================================================

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )


        # ====================================================
        # STEP 8
        # GENERATE ANSWER
        # ====================================================

        answer = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )


        return (
            answer,
            retrieved_documents,
            expanded_queries
        )


    except Exception as e:

        return (
            f"Error: {str(e)}",
            [],
            []
        )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main-title {

    font-size: 46px;

    font-weight: 700;

    margin-bottom: 30px;

}


.section-title {

    font-size: 32px;

    font-weight: 600;

}


.status-box {

    padding: 20px;

    border-radius: 10px;

    background-color: #f0f2f6;

    border: 1px solid #d8dbe2;

    margin-bottom: 25px;

}


.info-box {

    padding: 15px;

    border-radius: 8px;

    background-color: #eef5fb;

    margin-bottom: 15px;

}

</style>
""",

    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # SYSTEM INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "## System Information"
    )


    st.markdown(
        f"""
        <div class="info-box">

        <b>PDF Directory:</b>

        <br><br>

        {PDF_DIR}

        </div>
        """,

        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="info-box">

        <b>Database Directory:</b>

        <br><br>

        {DB_DIR}

        </div>
        """,

        unsafe_allow_html=True
    )


    st.divider()


    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    st.markdown(
        "## Controls"
    )


    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    st.write(
        "Upload PDF files"
    )


    uploaded_files = st.file_uploader(

        "Drag and drop PDF files here",

        type=[
            "pdf"
        ],

        accept_multiple_files=True

    )


    if uploaded_files:

        if st.button(
            "Save Uploaded PDFs",
            use_container_width=True
        ):

            saved = save_uploaded_files(
                uploaded_files
            )

            st.success(
                f"Saved {len(saved)} PDF file(s)."
            )


    # --------------------------------------------------------
    # RESET DATABASE
    # --------------------------------------------------------

    if st.button(
        "Reset Database",
        use_container_width=True
    ):

        success, message = (
            reset_database()
        )

        if success:

            st.success(
                message
            )

        else:

            st.error(
                message
            )


    # --------------------------------------------------------
    # INITIALIZE SYSTEM
    # --------------------------------------------------------

    if st.button(
        "Initialize System",
        use_container_width=True
    ):

        success, message = (
            initialize_system()
        )

        if success:

            st.success(
                message
            )

        else:

            st.error(
                message
            )


    # --------------------------------------------------------
    # PROCESS DOCUMENTS
    # --------------------------------------------------------

    if st.button(
        "Process Documents",
        use_container_width=True
    ):

        success, message = (
            process_documents()
        )

        if success:

            st.success(
                message
            )

        else:

            st.error(
                message
            )


# ============================================================
# MAIN PAGE
# ============================================================

st.markdown(
    """
    <div class="main-title">

    RAG System with Query Expansion

    </div>
    """,

    unsafe_allow_html=True
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown(
    "## System Status"
)


if not st.session_state.initialized:

    st.markdown(
        """
        <div class="status-box">

        <b>🔴 System not initialized</b>

        <br><br>

        Click <b>Initialize System</b>
        from the sidebar to initialize
        the RAG system.

        </div>
        """,

        unsafe_allow_html=True
    )

else:

    document_count = (
        get_document_count()
    )

    st.markdown(
        f"""
        <div class="status-box">

        <b>🟢 System initialized</b>

        <br><br>

        The RAG system is ready.

        <br><br>

        <b>Documents in database:</b>
        {document_count}

        </div>
        """,

        unsafe_allow_html=True
    )


# ============================================================
# QUERY INTERFACE
# ============================================================

st.markdown(
    "## Query Interface"
)


st.write(
    "Enter your question:"
)


question = st.text_input(

    "Question",

    placeholder=(
        "Ask a question about your "
        "uploaded documents..."
    ),

    label_visibility="collapsed"

)


# ============================================================
# NUMBER OF RESULTS
# ============================================================

number_of_results = st.slider(

    "Number of results to return",

    min_value=1,

    max_value=10,

    value=3

)


# ============================================================
# ASK QUESTION
# ============================================================

if st.button(
    "Ask Question",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )


    elif not st.session_state.initialized:

        st.error(
            "Please initialize the system first."
        )


    elif not st.session_state.documents_processed:

        st.error(
            "Please process your PDF documents first."
        )


    else:

        # ----------------------------------------------------
        # PROCESSING
        # ----------------------------------------------------

        with st.spinner(
            "Expanding query and searching documents..."
        ):

            (
                answer,
                documents,
                expanded_queries
            ) = answer_question(
                question
            )


        # ----------------------------------------------------
        # LIMIT DISPLAYED SOURCES
        # ----------------------------------------------------

        documents = documents[
            :number_of_results
        ]


        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.markdown(
            "## Answer"
        )

        st.write(
            answer
        )


        # ----------------------------------------------------
        # EXPANDED QUERIES
        # ----------------------------------------------------

        with st.expander(
            "View Expanded Queries"
        ):

            for i, query in enumerate(

                expanded_queries,

                1

            ):

                st.write(
                    f"{i}. {query}"
                )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        with st.expander(
            "View Retrieved Sources"
        ):

            if documents:

                for i, doc in enumerate(

                    documents,

                    1

                ):

                    source = (
                        doc.metadata.get(
                            "source",
                            "Unknown"
                        )
                    )


                    page = (
                        doc.metadata.get(
                            "page",
                            "Unknown"
                        )
                    )


                    st.markdown(
                        f"### Source {i}"
                    )


                    st.write(
                        f"**File:** {source}"
                    )


                    st.write(
                        f"**Page:** {page}"
                    )


                    st.write(
                        doc.page_content[:500]
                        + "..."
                    )


                    st.divider()


            else:

                st.write(
                    "No sources retrieved."
                )