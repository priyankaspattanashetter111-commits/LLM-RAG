import streamlit as st

from simple_rag import (
    generate_csv,
    EmbeddingModel,
    LLMModel,
    setup_chromadb,
    rag_pipeline,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Space Facts RAG",
    page_icon="🚀",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🚀 Space Facts RAG System"
)

st.write(
    "Ask questions about space using your local RAG system."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Model Configuration"
)


# ============================================================
# LLM SELECTION
# ============================================================

llm_type = st.sidebar.selectbox(
    "Select LLM Model:",
    [
        "llama3.2",
        "gemma3:4b",
        "qwen3:4b",
    ],
    format_func=lambda x: {
        "llama3.2": "Llama 3.2",
        "gemma3:4b": "Gemma 3 4B",
        "qwen3:4b": "Qwen 3 4B",
    }[x],
)


# ============================================================
# EMBEDDING SELECTION
# ============================================================

embedding_type = st.sidebar.selectbox(
    "Select Embedding Model:",
    [
        "nomic",
        "chroma",
    ],
    format_func=lambda x: {
        "nomic": (
            "Nomic Embed Text v2 MoE (Ollama)"
        ),
        "chroma": (
            "Chroma Default"
        ),
    }[x],
)


# ============================================================
# SHOW CURRENT MODEL
# ============================================================

st.sidebar.divider()

st.sidebar.write(
    "### Current Configuration"
)

st.sidebar.write(
    f"LLM: `{llm_type}`"
)

st.sidebar.write(
    f"Embedding: `{embedding_type}`"
)


# ============================================================
# CHECK WHETHER CONFIGURATION CHANGED
# ============================================================

configuration_changed = (
    "initialized" not in st.session_state
    or st.session_state.get("llm_type")
    != llm_type
    or st.session_state.get("embedding_type")
    != embedding_type
)


# ============================================================
# INITIALIZE RAG SYSTEM
# ============================================================

if configuration_changed:

    with st.spinner(
        "Loading models and space facts..."
    ):

        try:

            # ------------------------------------------------
            # Generate space facts
            # ------------------------------------------------

            facts = generate_csv()

            if facts is None:

                st.error(
                    "generate_csv() returned None. "
                    "Please make sure generate_csv() "
                    "contains 'return facts'."
                )

                st.stop()

            if not facts:

                st.error(
                    "No space facts were generated."
                )

                st.stop()

            # ------------------------------------------------
            # Store facts
            # ------------------------------------------------

            st.session_state.facts = facts

            # ------------------------------------------------
            # Initialize LLM
            # ------------------------------------------------

            st.session_state.llm_model = (
                LLMModel(
                    llm_type
                )
            )

            # ------------------------------------------------
            # Initialize embedding model
            # ------------------------------------------------

            st.session_state.embedding_model = (
                EmbeddingModel(
                    embedding_type
                )
            )

            # ------------------------------------------------
            # Extract document text
            # ------------------------------------------------

            documents = [
                fact["fact"]
                for fact in facts
            ]

            # ------------------------------------------------
            # Create IDs
            # ------------------------------------------------

            ids = [
                str(fact["id"])
                for fact in facts
            ]

            # ------------------------------------------------
            # Setup ChromaDB
            # ------------------------------------------------

            st.session_state.collection = (
                setup_chromadb(
                    documents,
                    st.session_state.embedding_model,
                    ids=ids,
                )
            )

            # ------------------------------------------------
            # Save configuration
            # ------------------------------------------------

            st.session_state.llm_type = (
                llm_type
            )

            st.session_state.embedding_type = (
                embedding_type
            )

            st.session_state.initialized = True

        except Exception as e:

            st.error(
                "Failed to initialize the RAG system."
            )

            st.exception(e)

            st.stop()


# ============================================================
# AVAILABLE SPACE FACTS
# ============================================================

with st.expander(
    "🪐 Available Space Facts"
):

    for fact in st.session_state.facts:

        st.write(
            f"- {fact['fact']}"
        )


# ============================================================
# DIVIDER
# ============================================================

st.divider()


# ============================================================
# QUESTION INPUT
# ============================================================

query = st.text_input(
    "Enter your question about space:",
    placeholder=(
        "e.g., What is the Hubble Space Telescope?"
    ),
)


# ============================================================
# GET ANSWER
# ============================================================

if st.button(
    "Get Answer",
    type="primary",
):

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "Searching facts and generating an answer..."
        ):

            try:

                # ------------------------------------------------
                # Run RAG pipeline
                # ------------------------------------------------

                response, references, augmented_prompt = (
                    rag_pipeline(
                        query,
                        st.session_state.collection,
                        st.session_state.llm_model,
                    )
                )

                # ------------------------------------------------
                # Create two columns
                # ------------------------------------------------

                col1, col2 = st.columns(2)

                # ------------------------------------------------
                # Response
                # ------------------------------------------------

                with col1:

                    st.markdown(
                        "### 🤖 Response"
                    )

                    st.write(
                        response
                    )

                # ------------------------------------------------
                # References
                # ------------------------------------------------

                with col2:

                    st.markdown(
                        "### 📖 References Used"
                    )

                    if references:

                        for ref in references:

                            st.write(
                                f"- {ref}"
                            )

                    else:

                        st.write(
                            "No references found."
                        )

                # ------------------------------------------------
                # Technical details
                # ------------------------------------------------

                with st.expander(
                    "🔍 Technical Details"
                ):

                    st.markdown(
                        "#### Augmented Prompt"
                    )

                    st.code(
                        augmented_prompt
                    )

                    st.markdown(
                        "#### Model Configuration"
                    )

                    st.write(
                        f"LLM: {llm_type}"
                    )

                    st.write(
                        f"Embedding: {embedding_type}"
                    )

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )

                st.exception(e)
