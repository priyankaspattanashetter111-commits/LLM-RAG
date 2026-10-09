import os
from typing import Optional

from dotenv import load_dotenv

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from newspaper import Article


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


class NewsArticleSummarizer:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_type: str = "ollama",
        model_name: str = "llama3.2",
    ):
        """
        Initialize the News Article Summarizer.

        Parameters:
            api_key:
                OpenAI API key.
                Not required when using Ollama.

            model_type:
                "ollama" -> Free local model
                "openai" -> OpenAI API model

            model_name:
                Ollama:
                    llama3.2
                    gemma3:4b
                    qwen3:4b

                OpenAI:
                    gpt-4o-mini
        """

        self.model_type = model_type
        self.model_name = model_name

        # ====================================================
        # INITIALIZE LANGUAGE MODEL
        # ====================================================

        if model_type == "openai":

            # Get API key from parameter or .env
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. "
                    "Add it to your .env file."
                )

            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=api_key,
            )

        elif model_type == "ollama":

            # Free local Ollama model
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
            )

        else:

            raise ValueError(
                "Unsupported model type. "
                "Use 'openai' or 'ollama'."
            )

        # ====================================================
        # TEXT SPLITTER
        # ====================================================

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )

    # ========================================================
    # FETCH ARTICLE
    # ========================================================

    def fetch_article(
        self,
        url: str
    ) -> Optional[Article]:

        """
        Fetch article content using newspaper3k.
        """

        try:

            print("\nDownloading article...")

            article = Article(url)

            article.download()
            article.parse()

            return article

        except Exception as e:

            print(
                f"Error fetching article: {e}"
            )

            return None

    # ========================================================
    # CREATE DOCUMENTS
    # ========================================================

    def create_documents(
        self,
        text: str
    ) -> list[Document]:

        """
        Split article text into smaller
        LangChain documents.
        """

        texts = self.text_splitter.split_text(text)

        docs = [
            Document(page_content=t)
            for t in texts
        ]

        return docs

    # ========================================================
    # SUMMARIZE ARTICLE
    # ========================================================

    def summarize(
        self,
        url: str,
        summary_type: str = "detailed"
    ) -> dict:

        """
        Main news article summarization pipeline.
        """

        # ----------------------------------------------------
        # STEP 1: FETCH ARTICLE
        # ----------------------------------------------------

        article = self.fetch_article(url)

        if not article:

            return {
                "error": "Failed to fetch article"
            }

        # ----------------------------------------------------
        # STEP 2: CREATE DOCUMENTS
        # ----------------------------------------------------

        docs = self.create_documents(
            article.text
        )

        print(
            f"\nArticle split into "
            f"{len(docs)} chunks."
        )

        # ----------------------------------------------------
        # STEP 3: DEFINE PROMPTS
        # ----------------------------------------------------

        if summary_type == "detailed":

            map_prompt_template = """
Write a detailed summary of the following text.

Focus on:
- Important facts
- Main events
- Key people
- Key organizations
- Important numbers
- Important conclusions

Do not add information that is not present
in the original text.

TEXT:
{text}

DETAILED SUMMARY:
"""

            combine_prompt_template = """
Write a detailed final summary of the following
summaries.

Combine the important information into one
coherent summary.

Remove unnecessary repetition.

Do not add information that is not present
in the provided summaries.

SUMMARIES:
{text}

FINAL DETAILED SUMMARY:
"""

        else:

            # Concise summary

            map_prompt_template = """
Write a concise summary of the following text.

Keep only the most important information.

Do not add information that is not present
in the original text.

TEXT:
{text}

CONCISE SUMMARY:
"""

            combine_prompt_template = """
Write a concise final summary using the
following summaries.

Keep the most important facts.

Remove unnecessary repetition.

Do not add information that is not present
in the provided summaries.

SUMMARIES:
{text}

FINAL CONCISE SUMMARY:
"""

        # ----------------------------------------------------
        # STEP 4: CREATE PROMPTS
        # ----------------------------------------------------

        map_prompt = PromptTemplate(
            template=map_prompt_template,
            input_variables=["text"]
        )

        combine_prompt = PromptTemplate(
            template=combine_prompt_template,
            input_variables=["text"]
        )

        # ----------------------------------------------------
        # STEP 5: CREATE MAP-REDUCE CHAIN
        # ----------------------------------------------------

        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )

        # ----------------------------------------------------
        # STEP 6: GENERATE SUMMARY
        # ----------------------------------------------------

        print(
            f"\nGenerating summary using:"
        )

        print(
            f"Model Type : {self.model_type}"
        )

        print(
            f"Model Name : {self.model_name}"
        )

        result = chain.invoke(docs)

        # ----------------------------------------------------
        # STEP 7: GET SUMMARY TEXT
        # ----------------------------------------------------

        if isinstance(result, dict):

            summary = result.get(
                "output_text",
                str(result)
            )

        else:

            summary = str(result)

        # ----------------------------------------------------
        # STEP 8: RETURN RESULT
        # ----------------------------------------------------

        return {
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "summary": summary,
            "url": url,
            "model_info": {
                "type": self.model_type,
                "name": self.model_name,
            },
        }


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # ========================================================
    # ARTICLE URL
    # ========================================================

    url = (
        "https://www.artificialintelligence-news.com/"
        "news/us-china-ai-chip-race/"
    )

    # ========================================================
    # FREE OLLAMA MODEL
    # ========================================================

    summarizer = NewsArticleSummarizer(
        model_type="ollama",
        model_name="llama3.2",
    )

    # ========================================================
    # GENERATE SUMMARY
    # ========================================================

    print("\nGenerating Llama Summary...")

    llama_summary = summarizer.summarize(
        url,
        summary_type="detailed"
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if "error" in llama_summary:

        print("\nERROR:")
        print(llama_summary["error"])

        return

    print("\n")
    print("=" * 70)

    print("NEWS ARTICLE SUMMARY")

    print("=" * 70)

    print("\nTitle:")
    print(llama_summary["title"])

    print("\nAuthors:")

    if llama_summary["authors"]:
        print(
            ", ".join(llama_summary["authors"])
        )
    else:
        print("Not available")

    print("\nPublished:")

    print(
        llama_summary["publish_date"]
    )

    print("\nModel:")

    print(
        f"{llama_summary['model_info']['type']} - "
        f"{llama_summary['model_info']['name']}"
    )

    print("\nSummary:")

    print("-" * 70)

    print(
        llama_summary["summary"]
    )

    print("\nURL:")

    print(
        llama_summary["url"]
    )

    print("=" * 70)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()