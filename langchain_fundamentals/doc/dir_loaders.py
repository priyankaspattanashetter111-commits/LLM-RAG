from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader


# ============================================================
# STEP 1: FIND CURRENT DOC FOLDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# STEP 2: CREATE DIRECTORY LOADER
# ============================================================

loader = DirectoryLoader(
    str(BASE_DIR),
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8"
    }
)


# ============================================================
# STEP 3: LOAD ALL TEXT FILES
# ============================================================

documents = loader.load()


# ============================================================
# STEP 4: DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("DIRECTORY LOADER")
print("=" * 60)

print("Directory:", BASE_DIR)

print("Number of documents:", len(documents))


for i, doc in enumerate(documents, start=1):

    print(f"\n--- Document {i} ---")

    print("Source:", doc.metadata.get("source"))

    print("\nContent:")

    print(doc.page_content[:500])