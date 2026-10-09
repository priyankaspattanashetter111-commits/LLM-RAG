import os
import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# =========================================
# 1. Load local embedding model
# =========================================

print("==== Loading embedding model ====")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================
# 2. Initialize ChromaDB
# =========================================

print("==== Starting ChromaDB ====")

chroma_client = chromadb.PersistentClient(
    path="./db/chroma_persistent_storage"
)

collection_name = "document_qa_collection"

collection = chroma_client.get_or_create_collection(
    name=collection_name
)


# =========================================
# 3. Load documents
# =========================================

def load_documents_from_directory(directory_path):

    print("==== Loading documents from directory ====")

    documents = []

    for filename in os.listdir(directory_path):

        if filename.endswith(".txt"):

            file_path = os.path.join(directory_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:

                documents.append({
                    "id": filename,
                    "text": file.read()
                })

    return documents


# =========================================
# 4. Split text into chunks
# =========================================

def split_text(text, chunk_size=1000, chunk_overlap=20):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start = end - chunk_overlap

    return chunks


# =========================================
# 5. Load documents
# =========================================

directory_path = "./data/new_articles"

documents = load_documents_from_directory(directory_path)


# =========================================
# 6. Create chunks
# =========================================

chunked_documents = []

for doc in documents:

    chunks = split_text(doc["text"])

    print("==== Splitting document into chunks ====")

    for i, chunk in enumerate(chunks):

        chunked_documents.append({
            "id": f"{doc['id']}_chunk{i+1}",
            "text": chunk
        })


# =========================================
# 7. Generate local embeddings
# =========================================

for doc in chunked_documents:

    print("==== Generating embedding ====")

    embedding = embedding_model.encode(
        doc["text"]
    ).tolist()

    doc["embedding"] = embedding


# =========================================
# 8. Store documents in ChromaDB
# =========================================

for doc in chunked_documents:

    print("==== Inserting chunk into ChromaDB ====")

    collection.upsert(

        ids=[doc["id"]],

        documents=[doc["text"]],

        embeddings=[doc["embedding"]]
    )


# =========================================
# 9. Query ChromaDB
# =========================================

def query_documents(question, n_results=2):

    print("==== Searching documents ====")

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(

        query_embeddings=[question_embedding],

        n_results=n_results
    )

    relevant_chunks = results["documents"][0]

    return relevant_chunks


# =========================================
# 10. Generate answer using Ollama
# =========================================

def generate_response(question, relevant_chunks):

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are an assistant for question-answering tasks.

Use the following context to answer the question.

If the answer is not present in the context,
say that you don't know.

Keep the answer concise and use a maximum of three sentences.

Context:

{context}

Question:

{question}
"""

    response = ollama.chat(

        model="llama3.2",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =========================================
# 11. Ask question
# =========================================

question = "Give me a brief overview of the articles. Be concise, please."

relevant_chunks = query_documents(question)

answer = generate_response(
    question,
    relevant_chunks
)


# =========================================
# 12. Print answer
# =========================================

print("\n==== Answer ====\n")

print(answer)