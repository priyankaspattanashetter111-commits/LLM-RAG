from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# ============================================================
# STEP 1: Define a prompt template
# ============================================================

prompt = ChatPromptTemplate.from_template(
    "Tell me a joke about {topic}"
)


# ============================================================
# STEP 2: Create the Ollama model
# ============================================================

model = ChatOllama(
    model="llama3.2:latest"
)


# ============================================================
# STEP 3: Create the chain
# ============================================================

chain = prompt | model | StrOutputParser()


# ============================================================
# STEP 4: Invoke the chain
# ============================================================

response = chain.invoke({"topic": "lions"})

print(response)