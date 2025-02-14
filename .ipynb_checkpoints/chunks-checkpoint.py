import os, sys
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

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
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return vectorstore

def retrieve_code_snippets(query, vectorstore, top_k=5):
    """Retrieve top-k relevant code snippets for a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    return retrieved_docs

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)

def generate_code_analysis(query, retrieved_docs):
    """Use GPT-4 to analyze retrieved code snippets."""
    code_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    messages = [
        SystemMessage(content="You are a senior AI engineer assisting in code analysis."),
        HumanMessage(content=f"Analyze the following code snippets and answer: {query}\n\n{code_context}")
    ]

    return llm(messages).content


if __name__ == "__main__":
    code_dir = sys.argv[1]  # Change this

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

        print("\nGenerating AI-powered insights...")
        response = generate_code_analysis(query, retrieved_docs)

        print("\n=== AI Response ===")
        print(response)

