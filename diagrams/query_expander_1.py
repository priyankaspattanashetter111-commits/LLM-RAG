
import os
from typing import List

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


load_dotenv()


# ============================================================
# QUERY EXPANDER
# ============================================================

class QueryExpander:
    """
    A class to expand a single query into multiple
    semantically similar queries to improve retrieval
    coverage.
    """

    def __init__(self, temperature: float = 0.3):

        """
        Initialize the QueryExpander.

        Args:
            temperature:
                Controls randomness in the LLM response.
                Lower values make the output more consistent.
        """

        # Local Ollama LLM
        self.llm = ChatOllama(
            model="llama3.2:latest",
            temperature=temperature
        )

        # Prompt template for query expansion
        self.query_expansion_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
Given the following question, generate 3 different
versions of the question that capture different
aspects and perspectives of the original question.

Make the variations semantically diverse but relevant.

Original Question:
{question}

Generate exactly 3 variations in the following format:

1. [First variation]
2. [Second variation]
3. [Third variation]

Do not add any explanation.
"""
        )


    def expand_query(
        self,
        question: str
    ) -> List[str]:

        """
        Expand a single query into multiple variations.

        Args:
            question:
                The original user question.

        Returns:
            List of query variations including
            the original question.
        """

        try:

            # Get variations from local LLM
            response = self.llm.invoke(
                self.query_expansion_prompt.format(
                    question=question
                )
            )

            # Get response text
            response_text = response.content.strip()

            # Parse numbered list from response
            variations = []

            for line in response_text.split("\n"):

                line = line.strip()

                if not line:
                    continue

                # Handle:
                # 1. query
                # 2. query
                # 3. query

                if ". " in line:

                    query = line.split(
                        ". ",
                        1
                    )[1].strip()

                    if query:
                        variations.append(query)

            # Add original question
            variations.append(question)

            return variations

        except Exception as e:

            print(
                f"Error in query expansion: {e}"
            )

            # If there is an error,
            # return only the original question

            return [question]


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # Initialize QueryExpander
    expander = QueryExpander()

    # Example questions
    questions = [

        "What are the main causes of global warming?",

        "How does exercise affect mental health?",

        "What are the benefits of renewable energy?"

    ]

    # Test query expansion
    for original_question in questions:

        print(
            f"\nOriginal Question: "
            f"{original_question}"
        )

        print(
            "Expanded Queries:"
        )

        expanded_queries = (
            expander.expand_query(
                original_question
            )
        )

        for i, query in enumerate(
            expanded_queries,
            1
        ):

            if query != original_question:

                print(
                    f"{i}. {query}"
                )

        print(
            f"Original: "
            f"{original_question}"
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

