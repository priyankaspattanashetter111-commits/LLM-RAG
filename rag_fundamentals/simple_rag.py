import csv
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "space_facts.csv"

DB_PATH = BASE_DIR / "chroma_db"


# ==========================================
# 1. SELECT MODELS
# ==========================================

def select_models():

    # ------------------------------
    # Select LLM model
    # ------------------------------

    print("\nSelect LLM Model:")
    print("1. Llama 3.2")
    print("2. Gemma 3 4B")
    print("3. Qwen 3 4B")

    while True:

        choice = input(
            "Enter choice (1-3): "
        ).strip()

        if choice == "1":

            llm_type = "llama3.2"

            break

        elif choice == "2":

            llm_type = "gemma3:4b"

            break

        elif choice == "3":

            llm_type = "qwen3:4b"

            break

        print("Please enter 1, 2, or 3")

    # ------------------------------
    # Select embedding model
    # ------------------------------

    print("\nSelect Embedding Model:")
    print("1. Chroma Default")
    print("2. Nomic Embed Text v2 MoE")

    while True:

        choice = input(
            "Enter choice (1 or 2): "
        ).strip()

        if choice == "1":

            embedding_type = "chroma"

            break

        elif choice == "2":

            embedding_type = "nomic"

            break

        print("Please enter 1 or 2")

    return llm_type, embedding_type


# ==========================================
# 2. GENERATE CSV FILE
# ==========================================

def generate_csv():

    facts = [

        {
            "id": 1,
            "fact": (
                "The first human to orbit Earth was "
                "Yuri Gagarin in 1961."
            )
        },

        {
            "id": 2,
            "fact": (
                "The Apollo 11 mission landed the first "
                "humans on the Moon in 1969."
            )
        },

        {
            "id": 3,
            "fact": (
                "The Hubble Space Telescope was launched "
                "in 1990 and has helped astronomers study "
                "the universe."
            )
        },

        {
            "id": 4,
            "fact": (
                "Mars has been explored by robotic "
                "orbiters, landers and rovers."
            )
        },

        {
            "id": 5,
            "fact": (
                "The International Space Station has "
                "supported continuous human presence "
                "since November 2000."
            )
        },

        {
            "id": 6,
            "fact": (
                "Voyager 1 was launched in 1977 and is "
                "the most distant human-made spacecraft."
            )
        },

        {
            "id": 7,
            "fact": (
                "SpaceX was founded in 2002 and develops "
                "reusable rockets and spacecraft."
            )
        },

        {
            "id": 8,
            "fact": (
                "The James Webb Space Telescope launched "
                "in December 2021 and observes the "
                "universe in infrared."
            )
        },

        {
            "id": 9,
            "fact": (
                "The Milky Way galaxy contains hundreds "
                "of billions of stars."
            )
        },

        {
            "id": 10,
            "fact": (
                "Black holes are regions of spacetime "
                "where gravity is so strong that light "
                "cannot escape from within the event horizon."
            )
        }
    ]

    with open(
        CSV_PATH,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["id", "fact"]
        )

        writer.writeheader()

        writer.writerows(facts)

    print(
        "\nCSV file 'space_facts.csv' "
        "created successfully!"
    )

    return facts


# ==========================================
# 3. LOAD CSV FILE
# ==========================================

def load_csv():

    documents = []

    ids = []

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    with open(
        CSV_PATH,
        mode="r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if (
                row.get("fact")
                and row["fact"].strip()
            ):

                ids.append(
                    str(row["id"])
                )

                documents.append(
                    row["fact"].strip()
                )

    print("\nLoaded documents:")

    for doc in documents:

        print(
            f" - {doc}"
        )

    return ids, documents


# ==========================================
# 4. EMBEDDING MODEL
# ==========================================

class EmbeddingModel:

    def __init__(
        self,
        model_type="nomic"
    ):

        self.model_type = model_type

        # ------------------------------
        # Chroma Default
        # ------------------------------

        if model_type == "chroma":

            self.embedding_function = (
                embedding_functions.DefaultEmbeddingFunction()
            )

        # ------------------------------
        # Nomic Embed Text v2 MoE
        # ------------------------------

        elif model_type == "nomic":

            self.embedding_function = (
                embedding_functions.OllamaEmbeddingFunction(
                    url=(
                        "http://localhost:11434/"
                        "api/embeddings"
                    ),
                    model_name=(
                        "nomic-embed-text-v2-moe:latest"
                    )
                )
            )

        else:

            raise ValueError(
                "Unsupported embedding model: "
                f"{model_type}"
            )

    def get_embedding_function(self):

        return self.embedding_function


# ==========================================
# 5. LLM MODEL
# ==========================================

class LLMModel:

    def __init__(
        self,
        model_type="llama3.2"
    ):

        self.model_type = model_type

        # ------------------------------
        # Ollama OpenAI-compatible API
        # ------------------------------

        self.client = OpenAI(
            base_url=(
                "http://localhost:11434/v1"
            ),
            api_key="ollama"
        )

        # ------------------------------
        # Available Ollama models
        # ------------------------------

        supported_models = {

            "llama3.2": "llama3.2:latest",

            "gemma3:4b": "gemma3:4b",

            "qwen3:4b": "qwen3:4b",

        }

        if model_type not in supported_models:

            raise ValueError(
                "Unsupported LLM model: "
                f"{model_type}"
            )

        self.model_name = (
            supported_models[model_type]
        )

    def generate_completion(
        self,
        messages
    ):

        try:

            response = (
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7
                )
            )

            return (
                response
                .choices[0]
                .message
                .content
            )

        except Exception as e:

            return (
                "Error generating response: "
                f"{str(e)}"
            )


# ==========================================
# 6. SET UP CHROMADB
# ==========================================

def setup_chromadb(
    documents,
    embedding_model,
    ids=None
):

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # ------------------------------
    # Delete old collection
    # ------------------------------

    try:

        client.delete_collection(
            "space_facts"
        )

    except Exception:

        pass

    # ------------------------------
    # Create collection
    # ------------------------------

    collection = client.create_collection(
        name="space_facts",
        embedding_function=(
            embedding_model
            .get_embedding_function()
        )
    )

    # ------------------------------
    # Generate IDs
    # ------------------------------

    if ids is None:

        ids = [
            str(i)
            for i in range(
                len(documents)
            )
        ]

    # ------------------------------
    # Add documents
    # ------------------------------

    if documents:

        collection.add(
            documents=documents,
            ids=ids
        )

    print(
        "\nDocuments added to "
        "ChromaDB collection successfully!"
    )

    return collection


# ==========================================
# 7. FIND RELATED CHUNKS
# ==========================================

def find_related_chunks(
    query,
    collection,
    top_k=2
):

    total_documents = (
        collection.count()
    )

    if total_documents == 0:

        print(
            "\nNo documents found "
            "in the collection."
        )

        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(
            top_k,
            total_documents
        ),
        include=[
            "documents",
            "metadatas"
        ]
    )

    retrieved_documents = (
        results["documents"][0]
    )

    metadata_results = (
        results.get("metadatas")
    )

    if (
        metadata_results
        and metadata_results[0]
    ):

        retrieved_metadata = (
            metadata_results[0]
        )

    else:

        retrieved_metadata = [
            {}
            for _ in retrieved_documents
        ]

    print(
        "\nRelated chunks found:"
    )

    for doc in retrieved_documents:

        print(
            f"- {doc}"
        )

    return list(
        zip(
            retrieved_documents,
            retrieved_metadata
        )
    )


# ==========================================
# 8. AUGMENT THE PROMPT
# ==========================================

def augment_prompt(
    query,
    related_chunks
):

    context = "\n".join(
        [
            chunk[0]
            for chunk in related_chunks
        ]
    )

    augmented_prompt = (
        f"Context:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    print(
        "\nAugmented prompt:"
    )

    print(
        augmented_prompt
    )

    return augmented_prompt


# ==========================================
# 9. COMPLETE RAG PIPELINE
# ==========================================

def rag_pipeline(
    query,
    collection,
    llm_model,
    top_k=2
):

    print(
        f"\nProcessing query: {query}"
    )

    related_chunks = (
        find_related_chunks(
            query,
            collection,
            top_k
        )
    )

    if not related_chunks:

        return (
            "No relevant documents were found.",
            [],
            ""
        )

    augmented_prompt = (
        augment_prompt(
            query,
            related_chunks
        )
    )

    response = (
        llm_model
        .generate_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Answer the question using "
                        "only the provided context. "
                        "If the context does not contain "
                        "the answer, say that the available "
                        "documents do not contain the answer. "
                        "Do not invent facts."
                    )
                },
                {
                    "role": "user",
                    "content": augmented_prompt
                }
            ]
        )
    )

    print(
        "\nGenerated response:"
    )

    print(
        response
    )

    references = [
        chunk[0]
        for chunk in related_chunks
    ]

    return (
        response,
        references,
        augmented_prompt
    )


# ==========================================
# 10. MAIN PROGRAM
# ==========================================

def main():

    print(
        "Starting the RAG pipeline demo..."
    )

    # ------------------------------
    # Select models
    # ------------------------------

    llm_type, embedding_type = (
        select_models()
    )

    # ------------------------------
    # Initialize LLM
    # ------------------------------

    llm_model = LLMModel(
        llm_type
    )

    # ------------------------------
    # Initialize embeddings
    # ------------------------------

    embedding_model = EmbeddingModel(
        embedding_type
    )

    print(
        f"\nUsing LLM: "
        f"{llm_type}"
    )

    print(
        f"Using Embeddings: "
        f"{embedding_type}"
    )

    # ------------------------------
    # Generate CSV
    # ------------------------------

    generate_csv()

    # ------------------------------
    # Load CSV
    # ------------------------------

    ids, documents = load_csv()

    # ------------------------------
    # Setup ChromaDB
    # ------------------------------

    collection = setup_chromadb(
        documents,
        embedding_model,
        ids
    )

    # ------------------------------
    # Sample queries
    # ------------------------------

    queries = [

        "What is the Hubble Space Telescope?",

        "Tell me about Mars exploration."

    ]

    # ------------------------------
    # Process queries
    # ------------------------------

    for query in queries:

        print(
            "\n" + "=" * 50
        )

        response, references, augmented_prompt = (
            rag_pipeline(
                query,
                collection,
                llm_model
            )
        )

        print(
            "\nFinal Results:"
        )

        print(
            "-" * 30
        )

        print(
            "Response:",
            response
        )

        print(
            "\nReferences used:"
        )

        for ref in references:

            print(
                f"- {ref}"
            )

        print(
            "=" * 50
        )


# ==========================================
# 11. RUN PROGRAM
# ==========================================

if __name__ == "__main__":

    main()
