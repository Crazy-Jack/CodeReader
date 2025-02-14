import os
import sys
import time
from rich.console import Console
from rich.markdown import Markdown
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Corrected imports

# Initialize Rich Console
console = Console()

def load_and_chunk_code(directory, chunk_size=500, chunk_overlap=50):
    """Load code files, split them into smaller chunks."""
    code_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())  # Fixed import
    return vectorstore

def retrieve_code_snippets(query, vectorstore, top_k=5):
    """Retrieve top-k relevant code snippets for a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    return retrieved_docs

def generate_code_analysis(query, retrieved_docs, llm):
    """Use GPT-4 to analyze retrieved code snippets and stream Markdown output character-by-character."""
    code_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    messages = [
        {"role": "system", "content": "You are a senior AI engineer assisting in code analysis."},
        {"role": "user", "content": f"Analyze the following code snippets and answer: {query}\n\n{code_context}"}
    ]

    response_stream = llm.stream(messages)
    
    console.print("\n[bold cyan]=== AI Response ===[/bold cyan]\n")

    markdown_buffer = ""  # Buffer to accumulate text
    for chunk in response_stream:
        text = chunk.content
        if text:
            for char in text:
                markdown_buffer += char  # Append new character
                console.clear()
                console.print(Markdown(markdown_buffer), soft_wrap=True)  # Render Markdown dynamically
                time.sleep(0.01)  # Simulate real-time streaming

    console.print("\n")  # Newline for better formatting

if __name__ == "__main__":
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3, streaming=True)  # Enable streaming

    if len(sys.argv) < 2:
        console.print("[bold red]Usage: python run.py <path_to_codebase>[/bold red]")
        sys.exit(1)

    code_dir = sys.argv[1]  # Get codebase path from command-line argument

    console.print("[bold green]Loading and chunking code files...[/bold green]")
    chunks = load_and_chunk_code(code_dir)

    console.print("[bold green]Storing embeddings...[/bold green]")
    vectorstore = store_code_embeddings(chunks)

    while True:
        query = console.input("\n[bold yellow]Enter your query about the codebase: [/bold yellow]").strip()

        # ✅ Skip if input is empty
        if not query:
            console.print("[dim]Skipping empty input...[/dim]")
            continue

        if query.lower() in ["exit", "quit"]:
            break

        console.print("\n[bold magenta]Retrieving relevant code snippets...[/bold magenta]")
        retrieved_docs = retrieve_code_snippets(query, vectorstore)

        console.print("\n[bold blue]Generating AI-powered insights (Streaming)...[/bold blue]")
        generate_code_analysis(query, retrieved_docs, llm)
