import chromadb
chroma_client = chromadb.Client()
from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()


collection_name = "test_collection"

collection = chroma_client.get_or_create_collection(collection_name, embedding_function=default_ef)

documents = [
    {"id": "doc1", "text": "Hello, World!"},
    {"id": "doc2", "text": "How are you today??"},
    {"id": "doc3", "text": "Goodbye. See you later!"}
    
]

for doc in documents:
    collection.upsert(
    ids=[doc["id"]],
    documents=[doc["text"]]
)
    


query_text = "Hello, World!"

results = collection.query(
    query_texts=[query_text],
    n_results=3,)

print(results)

for idx, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    for doc in documents:
        print(f" For the query: {query_text}, \n Found similar document: {document} (ID: {doc_id}, Distance: {distance})")

