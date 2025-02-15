import os
import sys
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live

console = Console()

# Initialize ChromaDB
vectorstore = Chroma(collection_name="code_vectors", embedding_function=OpenAIEmbeddings())

# Set chunking parameters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def load_and_chunk_code(directory):
    """Load code files, split them into smaller chunks, and store embeddings."""
    code_chunks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", ".js", ".cpp", ".java")):
                filepath = os.path.join(root, file)
                loader = TextLoader(filepath)
                documents = loader.load()
                chunks = text_splitter.split_documents(documents)
                code_chunks.extend(chunks)

    return code_chunks


def store_code_embeddings(chunks):
    """Embed and store code chunks in ChromaDB."""
    vectorstore.add_texts([chunk.page_content for chunk in chunks], metadatas=[{"source": chunk.metadata["source"]} for chunk in chunks])


def retrieve_code_snippets(query, top_k=5):
    """Retrieve top-k relevant code snippets for a query, ignoring unwanted folders."""
    all_docs = vectorstore.similarity_search(query, k=top_k)

    # ✅ Ignore results from unwanted folders
    ignored_folders = [".ipynb_checkpoints", "__pycache__", "node_modules"]
    filtered_docs = [
        doc for doc in all_docs
        if not any(ignored in doc.metadata.get("source", "") for ignored in ignored_folders)
    ]

    return filtered_docs
    
def load_and_encode_file(filepath):
    """Load, chunk, and encode a single file when modified or created."""
    if not filepath.endswith((".py", ".js", ".cpp", ".java")):
        return

    console.print(f"\n[bold yellow]Updating embeddings for {filepath}...[/bold yellow]")
    loader = TextLoader(filepath)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)

    # Remove old version of the file from vector store
    vectorstore.delete([filepath])

    # Add new chunks to the vector store
    vectorstore.add_texts([chunk.page_content for chunk in chunks], metadatas=[{"source": filepath} for chunk in chunks])

    console.print(f"[bold green]✅ Successfully updated vector database for {filepath}[/bold green]")


class FileChangeHandler(FileSystemEventHandler):
    """Handles file modifications and triggers vector updates."""
    def on_modified(self, event):
        if not event.is_directory:
            load_and_encode_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            load_and_encode_file(event.src_path)

    def on_deleted(self, event):
        """Handles file deletions in the vector database without blocking user input."""
        if not event.is_directory:
            file_path = event.src_path
            console.print(f"[bold yellow]🗑 File deleted: {file_path}[/bold yellow]")

            try:
                vectorstore.delete([file_path])  # Attempt to delete file embeddings
                console.print(f"[bold red]❌ Removed {file_path} from vector database[/bold red]")
            except Exception as e:
                console.print(f"[bold red]⚠️ Error deleting {file_path}: {e}[/bold red]")

            # ✅ Ensure control is returned to the user input loop immediately
            return


def start_monitoring(directory):
    """Start file monitoring with watchdog in a separate thread."""
    observer = Observer()
    event_handler = FileChangeHandler()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()

    console.print(f"\n[bold cyan]👀 Watching directory: {directory} for changes...[/bold cyan]\n")

    try:
        while True:
            time.sleep(1)  # Keep thread alive
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# def generate_code_analysis(query, retrieved_docs, llm):
#     """Use GPT-4 to analyze retrieved code snippets and stream output in a colorful Markdown format."""
    
#     console.print("\n[bold cyan]=== Retrieved Code Snippets ===[/bold cyan]\n")

#     # Display retrieved code snippets with syntax highlighting
#     for doc in retrieved_docs:
#         language = doc.metadata.get("source", "python")  # Default to Python
#         code = doc.page_content
#         console.print(Syntax(code, language, theme="monokai", line_numbers=True))

#     # Prepare query message
#     messages = [
#         {"role": "system", "content": "You are a senior AI engineer assisting in code analysis."},
#         {"role": "user", "content": f"Analyze the following code snippets and answer in Markdown format: {query}\n\n" + 
#                                     "\n\n".join([f"```{doc.metadata.get('source', 'python')}\n{doc.page_content}\n```" for doc in retrieved_docs])}
#     ]

#     console.print("\n[bold green]=== AI Response (Streaming) ===[/bold green]\n")

#     response_stream = llm.stream(messages)

#     markdown_text = ""

#     # Use Live to continuously update the Markdown output
#     with Live(auto_refresh=True, console=console) as live:
#         for chunk in response_stream:
#             text = chunk.content
#             if text:
#                 markdown_text += text  # Accumulate response
#                 live.update(Markdown(markdown_text))  # Render Markdown dynamically
#                 time.sleep(0.01)  # Simulate streaming delay

#     console.print("\n")  # Formatting for readability

def generate_code_analysis(query, retrieved_docs, llm):
    """Use GPT-4 to analyze retrieved code snippets and stream output in a colorful Markdown format."""
    
    console.print("\n[bold cyan]=== Retrieved Code Snippets ===[/bold cyan]\n")

    # ✅ Remove near-duplicate snippets before displaying
    seen_snippets = set()
    unique_docs = []
    for doc in retrieved_docs:
        normalized_code = doc.page_content.strip()
        if normalized_code not in seen_snippets:
            unique_docs.append(doc)
            seen_snippets.add(normalized_code)

    # ✅ Display only unique snippets with filename and separator line
    for i, doc in enumerate(unique_docs):
        language = doc.metadata.get("source", "python")  # Default to Python
        code = doc.page_content
        filename = doc.metadata.get("source", "Unknown File")  # Get file path

        # ✅ Display the filename above the code snippet
        console.print(f"[bold magenta]File: {filename}[/bold magenta]")
        console.print(Syntax(code, language, theme="monokai", line_numbers=True))

        # ✅ Add a separator between snippets (except after the last one)
        if i < len(unique_docs) - 1:
            console.print("[bold yellow]──────────────────────────────────────────────────────[/bold yellow]")

    # ✅ Generate a cleaner prompt
    unique_code_blocks = "\n\n".join([
        f"**File: {doc.metadata.get('source', 'Unknown File')}**\n\n```{doc.metadata.get('source', 'python')}\n{doc.page_content}\n```"
        for doc in unique_docs
    ])

    messages = [
        {"role": "system", "content": "You are a senior AI engineer specializing in code analysis. Your task is to analyze and explain unique insights."},
        {"role": "user", "content": f"Analyze the following unique code snippets and answer in Markdown format:\n\n{unique_code_blocks}\n\n"
                                    "### Important Notes:\n"
                                    "- Avoid repeating explanations.\n"
                                    "- Summarize common patterns if multiple snippets are similar.\n"
                                    "- Provide only the key insights.\n"
                                    "- Also provide code solutions to fix the update, suggest in seperate code blocks"}
    ]

    console.print("\n[bold green]=== AI Response (Streaming) ===[/bold green]\n")

    response_stream = llm.stream(messages)

    markdown_text = ""  # ✅ Start with an empty Markdown buffer

    # ✅ Use Live to continuously update the Markdown output without duplication
    with Live(console=console, refresh_per_second=10) as live:
        for chunk in response_stream:
            text = chunk.content
            if text:
                markdown_text += text  # ✅ Accumulate new content
                live.update(Markdown(markdown_text))  # ✅ Update Markdown content only with new data
                time.sleep(0.01)  # ✅ Controlled streaming delay to simulate real-time flow

    console.print("\n")  # ✅ Final formatting for readability


def question_answering_loop():
    """Handles user queries in a separate thread while file monitoring continues."""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3, streaming=True)  # Enable streaming

    while True:
        query = input("\nEnter your query about the codebase: ").strip()  # ✅ Trim whitespace

        # ✅ Skip empty queries
        if not query:
            continue
        
        if query.lower() in ["exit", "quit"]:
            console.print("\n[bold red]Exiting question-answering process...[/bold red]")
            break

        print("\nRetrieving relevant code snippets...")
        retrieved_docs = retrieve_code_snippets(query)

        print("\nGenerating AI-powered insights (Streaming)...")
        generate_code_analysis(query, retrieved_docs, llm)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <path_to_codebase>")
        sys.exit(1)

    code_dir = sys.argv[1]  # Get codebase path from command-line argument

    print("Loading and chunking code files...")
    chunks = load_and_chunk_code(code_dir)

    print("Storing embeddings...")
    store_code_embeddings(chunks)

    # Start file monitoring in a separate thread
    monitoring_thread = threading.Thread(target=start_monitoring, args=(code_dir,), daemon=True)
    monitoring_thread.start()

    # Start the question-answering loop
    question_answering_loop()
