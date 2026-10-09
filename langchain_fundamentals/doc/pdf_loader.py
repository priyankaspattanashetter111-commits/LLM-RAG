
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


# ============================================================
# STEP 1: GET THE PDF FILE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

pdf_path = BASE_DIR / "Major_Project_Report (2).pdf"


# ============================================================
# STEP 2: LOAD THE PDF
# ============================================================

loader = PyPDFLoader(str(pdf_path))

documents = loader.load()


# ============================================================
# STEP 3: DISPLAY PDF INFORMATION
# ============================================================

print("=" * 60)
print("STEP 1: PDF LOADED")
print("=" * 60)

print("PDF file:", pdf_path.name)

print("Total number of pages:", len(documents))


# ============================================================
# STEP 4: DISPLAY EACH PAGE
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: PDF PAGE CONTENT")
print("=" * 60)

for i, doc in enumerate(documents, start=1):

    print(f"\n--- Page {i} ---")

    print("Source:", doc.metadata.get("source"))

    print("Page number:", doc.metadata.get("page"))

    print("\nContent:")

    print(doc.page_content)


# ============================================================
# STEP 5: DISPLAY COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PDF LOADING COMPLETED SUCCESSFULLY!")
print("=" * 60)