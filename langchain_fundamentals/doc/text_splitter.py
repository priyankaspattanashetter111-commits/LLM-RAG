from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
I have a dream that my four little children will one day live
in a nation where they will not be judged by the color of their
skin but by the content of their character.

This is a famous speech about equality and freedom.
It expresses the hope that people will be treated fairly.
"""



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)



chunks = text_splitter.split_text(text)



print("=" * 60)
print("TEXT SPLITTER")
print("=" * 60)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):

    print(f"\n--- Chunk {i} ---")
    print(chunk)
    print("Length:", len(chunk))