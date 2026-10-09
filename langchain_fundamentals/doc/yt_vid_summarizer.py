
import os
import yt_dlp
import whisper

from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


class YoutubeVideoSummarizer:

    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
    ):
        """
        Initialize all local/free AI models.
        """

        # ---------------------------------------------------------
        # WHISPER
        # ---------------------------------------------------------
        print("\nLoading Whisper model...")
        print("Using Whisper: tiny (faster CPU transcription)")

        self.whisper_model = whisper.load_model("tiny")

        # ---------------------------------------------------------
        # LLM
        # ---------------------------------------------------------
        print(f"Loading LLM: {llm_model_name}")

        self.llm = ChatOllama(
            model=llm_model_name,
            temperature=0.2,
        )

        # ---------------------------------------------------------
        # EMBEDDINGS
        # ---------------------------------------------------------
        print(f"Loading Embedding Model: {embedding_model_name}")

        self.embedding_model = OllamaEmbeddings(
            model=embedding_model_name
        )

        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name

    # =============================================================
    # DOWNLOAD YOUTUBE VIDEO
    # =============================================================

    def download_video(self, url: str):

        print("\nDownloading YouTube video/audio...")

        os.makedirs("downloads", exist_ok=True)

        ydl_opts = {
            "format": "bestaudio/best",

            "outtmpl": "downloads/%(title)s.%(ext)s",

            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }
            ],

            "quiet": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            video_title = info.get(
                "title",
                "Unknown Title"
            )

            original_path = ydl.prepare_filename(info)

            audio_path = (
                os.path.splitext(original_path)[0]
                + ".mp3"
            )

        print(f"\nVideo Title: {video_title}")
        print(f"Audio Path: {audio_path}")

        return audio_path, video_title

    # =============================================================
    # TRANSCRIBE AUDIO
    # =============================================================

    def transcribe_audio(self, audio_path: str) -> str:

        print("\nTranscribing audio using Whisper...")
        print("Whisper tiny is optimized for CPU speed.")
        print("This may still take some time depending on video length...")

        result = self.whisper_model.transcribe(
            audio_path,

            # Faster CPU transcription
            fp16=False,

            # English videos are common for this project
            language="en",

            # Prevent unnecessary translation
            task="transcribe",

            # Reduce extra processing
            verbose=False,
        )

        transcript = result["text"]

        print("\nTranscription completed.")

        return transcript

    # =============================================================
    # CREATE DOCUMENT CHUNKS
    # =============================================================

    def create_documents(
        self,
        text: str,
        video_title: str
    ) -> List[Document]:

        print("\nCreating transcript chunks...")

        text_splitter = RecursiveCharacterTextSplitter(

            # Larger chunks = fewer chunks = fewer LLM calls
            chunk_size=2500,

            chunk_overlap=150,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
        )

        chunks = text_splitter.split_text(text)

        documents = []

        for index, chunk in enumerate(chunks):

            document = Document(
                page_content=chunk,

                metadata={
                    "source": video_title,
                    "chunk": index,
                }
            )

            documents.append(document)

        print(
            f"Created {len(documents)} document chunks."
        )

        return documents

    # =============================================================
    # CREATE CHROMADB
    # =============================================================

    def create_vector_store(
        self,
        documents: List[Document]
    ):

        print("\nCreating ChromaDB vector store...")

        # Use a unique collection name for this run
        collection_name = "youtube_summarizer"

        vector_store = Chroma.from_documents(

            documents=documents,

            embedding=self.embedding_model,

            collection_name=collection_name,

            persist_directory="./chroma_db",
        )

        print("ChromaDB created successfully.")

        return vector_store

    # =============================================================
    # FAST SUMMARY
    # =============================================================

    def generate_summary(
        self,
        documents: List[Document]
    ) -> str:

        print("\nGenerating video summary...")

        # ---------------------------------------------------------
        # GROUP DOCUMENTS
        # ---------------------------------------------------------

        # Instead of summarizing every chunk individually,
        # combine multiple chunks together.

        group_size = 5

        groups = []

        for i in range(
            0,
            len(documents),
            group_size
        ):

            group = documents[
                i:i + group_size
            ]

            group_text = "\n\n".join(
                document.page_content
                for document in group
            )

            groups.append(group_text)

        print(
            f"Created {len(groups)} summary groups."
        )

        # ---------------------------------------------------------
        # SUMMARIZE EACH GROUP
        # ---------------------------------------------------------

        group_summaries = []

        for index, group in enumerate(groups):

            print(
                f"Summarizing group "
                f"{index + 1}/{len(groups)}..."
            )

            prompt = f"""
You are an expert YouTube video summarizer.

Summarize the following transcript section.

Focus on:

- Main ideas
- Important facts
- Important examples
- Key arguments
- Technical concepts
- Conclusions

Be concise but informative.

Do not add information that is not present
in the transcript.

Transcript:

{group}

Summary:
"""

            response = self.llm.invoke(prompt)

            group_summaries.append(
                response.content
            )

        # ---------------------------------------------------------
        # FINAL SUMMARY
        # ---------------------------------------------------------

        print("\nCreating final summary...")

        combined_summaries = "\n\n".join(
            group_summaries
        )

        final_prompt = f"""
You are an expert YouTube video summarizer.

Below are summaries from different sections
of a YouTube video.

Create one clear and useful final summary.

Structure the answer as:

1. Main Topic
2. Key Points
3. Important Details
4. Examples
5. Main Conclusion
6. Practical / Actionable Points

Avoid unnecessary repetition.

Do not add information that is not contained
in the section summaries.

Section summaries:

{combined_summaries}

Final Summary:
"""

        final_response = self.llm.invoke(
            final_prompt
        )

        return final_response.content

    # =============================================================
    # QUESTION ANSWERING
    # =============================================================

    def answer_question(
        self,
        vector_store,
        question: str
    ) -> str:

        print("\nSearching the video...")

        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 4
            }
        )

        relevant_documents = retriever.invoke(
            question
        )

        context = "\n\n".join(
            document.page_content
            for document in relevant_documents
        )

        prompt = f"""
You are answering a question about a YouTube video.

Use ONLY the information from the video transcript
provided below.

If the answer cannot be found in the provided context,
say exactly:

"I couldn't find that information in the video."

Do not make up information.

Video transcript context:

{context}

Question:

{question}

Answer:
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content

    # =============================================================
    # PROCESS VIDEO
    # =============================================================

    def process_video(
        self,
        url: str
    ) -> Dict:

        try:

            # -----------------------------------------------------
            # STEP 1: DOWNLOAD
            # -----------------------------------------------------

            audio_path, video_title = (
                self.download_video(url)
            )

            # -----------------------------------------------------
            # STEP 2: TRANSCRIPTION
            # -----------------------------------------------------

            transcript = self.transcribe_audio(
                audio_path
            )

            # -----------------------------------------------------
            # STEP 3: DOCUMENTS
            # -----------------------------------------------------

            documents = self.create_documents(
                transcript,
                video_title
            )

            # -----------------------------------------------------
            # STEP 4: CHROMADB
            # -----------------------------------------------------

            vector_store = self.create_vector_store(
                documents
            )

            # -----------------------------------------------------
            # STEP 5: SUMMARY
            # -----------------------------------------------------

            summary = self.generate_summary(
                documents
            )

            # -----------------------------------------------------
            # DELETE TEMP AUDIO
            # -----------------------------------------------------

            if os.path.exists(audio_path):

                os.remove(audio_path)

                print(
                    "\nTemporary audio file deleted."
                )

            return {
                "title": video_title,

                "summary": summary,

                "transcript": transcript,

                "vector_store": vector_store,
            }

        except Exception as e:

            print(
                "\nERROR PROCESSING VIDEO:"
            )

            print(str(e))

            return None


# =================================================================
# MAIN PROGRAM
# =================================================================

def main():

    print("\n")

    print("=" * 65)
    print("          FREE YOUTUBE VIDEO SUMMARIZER")
    print("=" * 65)

    # -------------------------------------------------------------
    # LLM SELECTION
    # -------------------------------------------------------------

    print("\nAvailable FREE LLM Models:")

    print("1. Llama 3.2")
    print("2. Gemma 3 4B")
    print("3. Qwen 3 4B")

    while True:

        llm_choice = input(
            "\nChoose LLM model (1/2/3): "
        ).strip()

        if llm_choice == "1":

            llm_model = "llama3.2:latest"

            break

        elif llm_choice == "2":

            llm_model = "gemma3:4b"

            break

        elif llm_choice == "3":

            llm_model = "qwen3:4b"

            break

        else:

            print(
                "Invalid choice. "
                "Please choose 1, 2, or 3."
            )

    # -------------------------------------------------------------
    # EMBEDDING SELECTION
    # -------------------------------------------------------------

    print("\nAvailable FREE Embedding Models:")

    print("1. Nomic Embed Text")
    print("2. Nomic Embed Text v2")
    print("3. All-MiniLM")

    while True:

        embedding_choice = input(
            "\nChoose embedding model (1/2/3): "
        ).strip()

        if embedding_choice == "1":

            embedding_model = (
                "nomic-embed-text:latest"
            )

            break

        elif embedding_choice == "2":

            embedding_model = (
                "nomic-embed-text-v2-moe:latest"
            )

            break

        elif embedding_choice == "3":

            embedding_model = (
                "all-minilm:latest"
            )

            break

        else:

            print(
                "Invalid choice. "
                "Please choose 1, 2, or 3."
            )

    # -------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------

    print("\n")

    print("=" * 65)
    print("CURRENT CONFIGURATION")
    print("=" * 65)

    print(
        f"LLM        : {llm_model}"
    )

    print(
        f"Embeddings : {embedding_model}"
    )

    print(
        "Whisper    : tiny"
    )

    print(
        "Vector DB  : ChromaDB"
    )

    print(
        "Cost       : FREE / LOCAL"
    )

    print("=" * 65)

    # -------------------------------------------------------------
    # INITIALIZE
    # -------------------------------------------------------------

    try:

        summarizer = YoutubeVideoSummarizer(

            llm_model_name=llm_model,

            embedding_model_name=embedding_model,
        )

    except Exception as e:

        print(
            "\nCould not initialize models."
        )

        print(
            f"Error: {str(e)}"
        )

        return

    # -------------------------------------------------------------
    # URL
    # -------------------------------------------------------------

    url = input(
        "\nEnter YouTube URL: "
    ).strip()

    if not url:

        print(
            "\nNo YouTube URL provided."
        )

        return

    # -------------------------------------------------------------
    # PROCESS VIDEO
    # -------------------------------------------------------------

    result = summarizer.process_video(
        url
    )

    if result is None:

        print(
            "\nVideo processing failed."
        )

        return

    # -------------------------------------------------------------
    # DISPLAY TITLE
    # -------------------------------------------------------------

    print("\n")

    print("=" * 65)
    print("VIDEO TITLE")
    print("=" * 65)

    print(
        result["title"]
    )

    # -------------------------------------------------------------
    # DISPLAY SUMMARY
    # -------------------------------------------------------------

    print("\n")

    print("=" * 65)
    print("VIDEO SUMMARY")
    print("=" * 65)

    print(
        result["summary"]
    )

    # -------------------------------------------------------------
    # Q&A
    # -------------------------------------------------------------

    print("\n")

    print("=" * 65)
    print("VIDEO Q&A")
    print("=" * 65)

    print(
        "\nYou can now ask questions about the video."
    )

    print(
        "Type 'quit' to exit."
    )

    while True:

        question = input(
            "\nYour question: "
        ).strip()

        if question.lower() == "quit":

            print(
                "\nThank you! Exiting..."
            )

            break

        if not question:

            continue

        try:

            answer = summarizer.answer_question(

                result["vector_store"],

                question
            )

            print("\nAnswer:")

            print(
                answer
            )

        except Exception as e:

            print(
                f"\nError answering question: {str(e)}"
            )


# =================================================================
# RUN PROGRAM
# =================================================================

if __name__ == "__main__":

    main()

