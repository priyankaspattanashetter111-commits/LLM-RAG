from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 1. LIST OF DOCUMENTS / URLs
# ============================================================

documents = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-servers-discord/",
    "https://beebom.com/how-list-groups-linux/",
    "https://beebom.com/how-open-port-linux/",
    "https://beebom.com/linux-vs-windows/",
]


# ============================================================
# 2. FREE LOCAL LLM MODEL
# ============================================================

model_name = "llama3.2"


# ============================================================
# 3. SCRAPE DOCUMENTS
# ============================================================

def scrape_docs(urls: List[str]) -> List[Dict]:
    """Scrape content from URLs using SeleniumURLLoader"""

    try:
        loader = SeleniumURLLoader(urls=urls)

        raw_docs = loader.load()

        print(f"\nSuccessfully loaded {len(raw_docs)} documents")

        # Print information about loaded documents
        for doc in raw_docs:
            print(
                f"\nSource: {doc.metadata.get('source', 'No source')}"
            )

            print(
                f"Content length: {len(doc.page_content)} characters"
            )

        return raw_docs

    except Exception as e:

        print(
            f"Error during document loading: {str(e)}"
        )

        return []


# ============================================================
# 4. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def split_documents(pages_content: List[Dict]) -> tuple:
    """Split documents into smaller chunks"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    all_texts = []
    all_metadatas = []

    for document in pages_content:

        # Extract text from Document object
        text = document.page_content

        # Get source URL
        source = document.metadata.get(
            "source",
            ""
        )

        # Split text into chunks
        chunks = text_splitter.split_text(text)

        # Store chunks and metadata
        for chunk in chunks:

            all_texts.append(chunk)

            all_metadatas.append(
                {
                    "source": source
                }
            )

    print(
        f"Created {len(all_texts)} chunks of text"
    )

    return all_texts, all_metadatas


# ============================================================
# 5. CREATE VECTOR STORE
# ============================================================

def create_vector_store(
    texts: List[str],
    metadatas: List[Dict]
):
    """Create vector store using ChromaDB"""

    # FREE LOCAL EMBEDDING MODEL
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Create Chroma vector database
    db = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        collection_name="beebom_documents"
    )

    return db


# ============================================================
# 6. SETUP QA CHAIN
# ============================================================

def setup_qa_chain(db):

    """Set up QA chain with polite response template"""

    # FREE LOCAL CHAT MODEL
    llm = ChatOllama(
        model=model_name,
        temperature=0
    )

    # Create retriever
    retriever = db.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    # Custom prompt
    prompt = ChatPromptTemplate.from_template(
        """
Please provide a polite and helpful response
to the following question.

Use ONLY the information available in the context.

If the answer is not available in the context,
say that you don't know based on the provided
information.

### Context:

{context}

### Question:

{question}

### Answer:
"""
    )

    # Create RAG chain
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ============================================================
# 7. PROCESS USER QUERY
# ============================================================

def process_query(
    chain_and_retriever,
    query: str
):

    """Process a query and return response"""

    try:

        # Unpack chain and retriever
        chain, retriever = chain_and_retriever

        # Generate answer
        response = chain.invoke(query)

        # Retrieve relevant documents
        docs = retriever.invoke(query)

        # Get source URLs
        sources = []

        for doc in docs:

            source = doc.metadata.get(
                "source",
                ""
            )

            if source and source not in sources:
                sources.append(source)

        sources_str = ", ".join(sources)

        return {
            "answer": response,
            "sources": sources_str
        }

    except Exception as e:

        print(
            f"Error processing query: {str(e)}"
        )

        return {
            "answer": (
                "I apologize, but I encountered "
                "an error while processing your question."
            ),
            "sources": ""
        }


# ============================================================
# 8. MAIN FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # Step 1: Scrape documents
    # --------------------------------------------------------

    print("\nScraping documents...")

    pages_content = scrape_docs(documents)

    if not pages_content:

        print(
            "No documents were loaded."
        )

        return

    # --------------------------------------------------------
    # Step 2: Split documents
    # --------------------------------------------------------

    print("\nSplitting documents...")

    all_texts, all_metadatas = split_documents(
        pages_content
    )

    # --------------------------------------------------------
    # Step 3: Create vector store
    # --------------------------------------------------------

    print("\nCreating vector store...")

    db = create_vector_store(
        all_texts,
        all_metadatas
    )

    # --------------------------------------------------------
    # Step 4: Setup QA chain
    # --------------------------------------------------------

    print("\nSetting up QA chain...")

    qa_chain = setup_qa_chain(db)

    # --------------------------------------------------------
    # Step 5: Interactive question-answer loop
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "RAG QA Bot is ready!"
    )

    print(
        "Type 'quit' to exit."
    )

    print(
        "========================================"
    )

    while True:

        query = input(
            "\nEnter your question: "
        ).strip()

        # Exit
        if query.lower() == "quit":
            print("\nGoodbye!")
            break

        # Ignore empty input
        if not query:
            print(
                "Please enter a question."
            )
            continue

        # Process question
        result = process_query(
            qa_chain,
            query
        )

        # Print answer
        print(
            "\nAnswer:"
        )

        print(
            result["answer"]
        )

        # Print sources
        print(
            "\nSources:"
        )

        print(
            result["sources"]
        )


# ============================================================
# 9. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()