from pathlib import Path
import re

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# STEP 1: Load document
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "dream.txt"

loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

documents = loader.load()

print("=" * 60)
print("STEP 1: DOCUMENT LOADED")
print("=" * 60)

print("Number of documents:", len(documents))


# ============================================================
# STEP 2: Clean text
# ============================================================

def clean_text(text):
    """
    Clean unnecessary whitespace and newlines.
    """

    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # Replace 3 or more newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


for doc in documents:
    doc.page_content = clean_text(doc.page_content)


print("\n" + "=" * 60)
print("STEP 2: TEXT CLEANED")
print("=" * 60)

print(documents[0].page_content[:500])


# ============================================================
# STEP 3: Split text
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


print("\n" + "=" * 60)
print("STEP 3: TEXT SPLITTING")
print("=" * 60)

print("Number of chunks:", len(chunks))


for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i} ---")

    print(chunk.page_content)


# ============================================================
# STEP 4: Create HuggingFace embeddings
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: EMBEDDINGS")
print("=" * 60)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# ============================================================
# STEP 5: Create FAISS VectorStore
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: FAISS VECTOR STORE")
print("=" * 60)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector store created successfully!")


# ============================================================
# STEP 6: Create Retriever
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: RETRIEVER")
print("=" * 60)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 2
    }
)

print("Retriever created successfully!")


# ============================================================
# STEP 7: Query
# ============================================================

query = "What is the dream about?"

print("\n" + "=" * 60)
print("STEP 7: QUERY")
print("=" * 60)

print("Query:", query)


# ============================================================
# STEP 8: Retrieve relevant documents
# ============================================================

retrieved_docs = retriever.invoke(query)


# ============================================================
# STEP 9: Display retrieved documents
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: RETRIEVED DOCUMENTS")
print("=" * 60)

print(
    "Number of retrieved documents:",
    len(retrieved_docs)
)


for i, doc in enumerate(retrieved_docs, start=1):

    print(f"\n--- Retrieved Document {i} ---")

    print(
        "Source:",
        doc.metadata.get("source")
    )

    print("\nContent:")

    print(doc.page_content)


# ============================================================
# STEP 10: Create Chat Prompt
# ============================================================
# Lesson 219:
# Simple RAG System with Chat and LangChain Chains
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: CHAT PROMPT")
print("=" * 60)


prompt = ChatPromptTemplate.from_template(
    """
    Please use the following documents to answer the question.

    Documents:
    {docs}

    Question:
    {question}

    Answer the question based only on the provided documents.
    If the answer is not available in the documents,
    say "I could not find the answer in the provided documents."
    """
)


print("Chat prompt created successfully!")


# ============================================================
# STEP 11: Create Chat Model
# ============================================================
# Using Ollama instead of OpenAI.
# This runs your Llama model locally.
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: CHAT MODEL")
print("=" * 60)


model = ChatOllama(
    model="llama3.2:latest"
)


print("Ollama Llama 3.2 model ready!")


# ============================================================
# STEP 12: Create LangChain RAG Chain
# ============================================================

print("\n" + "=" * 60)
print("STEP 12: LANGCHAIN RAG CHAIN")
print("=" * 60)


chain = prompt | model | StrOutputParser()


print("RAG chain created successfully!")


# ============================================================
# STEP 13: Prepare Retrieved Documents
# ============================================================

docs_text = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)


# ============================================================
# STEP 14: Generate Final Answer
# ============================================================

print("\n" + "=" * 60)
print("STEP 13: GENERATING ANSWER")
print("=" * 60)


response = chain.invoke({
    "docs": docs_text,
    "question": query
})


# ============================================================
# STEP 15: Display Final Answer
# ============================================================

print("\n" + "=" * 60)
print("FINAL RAG ANSWER")
print("=" * 60)

print(response)