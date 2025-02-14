import os
import sys
import time
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Corrected imports

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
    """Use GPT-4 to analyze retrieved code snippets and stream output."""
    code_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    messages = [
        {"role": "system", "content": "You are a senior AI engineer assisting in code analysis."},
        {"role": "user", "content": f"Analyze the following code snippets and answer: {query}\n\n{code_context}"}
    ]

    # Stream the response
    response_stream = llm.stream(messages)
    
    print("\n=== AI Response ===")
    for chunk in response_stream:
        text = chunk.content
        if text:
            for char in text:
                print(char, end="", flush=True)  # Print character-by-character
                time.sleep(0.01)  # Simulate real-time streaming
    print("\n")  # Newline for better formatting

if __name__ == "__main__":
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3, streaming=True)  # Enable streaming

    if len(sys.argv) < 2:
        print("Usage: python run.py <path_to_codebase>")
        sys.exit(1)

    code_dir = sys.argv[1]  # Get codebase path from command-line argument

    print("Loading and chunking code files...")
    chunks = load_and_chunk_code(code_dir)

    print("Storing embeddings...")
    vectorstore = store_code_embeddings(chunks)

    while True:
        query = input("\nEnter your query about the codebase: ")
        if query.lower() in ["exit", "quit"]:
            break

        print("\nRetrieving relevant code snippets...")
        retrieved_docs = retrieve_code_snippets(query, vectorstore)

        print("\nGenerating AI-powered insights (Streaming)...")
        generate_code_analysis(query, retrieved_docs, llm)
