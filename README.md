# **AI-Powered Codebase Query System** 🚀  
> **Interact with your codebase using GPT-4 for real-time AI-powered insights!**  
> Search, retrieve, and analyze code snippets with dynamic Markdown rendering.

---

## **✨ Features**
👉 **Vector-Based Code Search** – Uses **ChromaDB** and **OpenAIEmbeddings** to store and retrieve relevant code snippets.  
👉 **Live AI-Powered Code Analysis** – GPT-4 provides contextual explanations for retrieved snippets.  
👉 **Real-Time Markdown Rendering** – AI-generated insights are streamed **character-by-character** and rendered in the terminal.  
👉 **Automatic Codebase Updates** – Detects file changes and **automatically updates the vector database** in real-time.  

---

## **📦 Installation & Setup**

### **1⃣ Configure the Environment**
```bash
conda create -n code_reader python=3.10 -y
conda activate code_reader

pip install langchain openai chromadb tiktoken faiss-cpu
pip install -U langchain-community
pip install langchain_openai
pip install rich
pip install watchdog
```

### **2⃣ Set Up Your OpenAI API Key**
```bash
export OPENAI_API_KEY='your-open-ai-api-key'
```

---

## **🚀 Usage**
### **Interact with Your Codebase via CLI**
Run the script and specify your **codebase root folder**:
```bash
python3 run.py [your-codebase-root-folder]
```

- Once started, the tool will **monitor file changes** and **update embeddings automatically**.  
- You can **ask AI-powered questions** about your code in real-time!  

---

## **💡 How It Works**
1. **Embeds your code** into a vector database using **ChromaDB**.  
2. **Monitors file changes** and updates stored embeddings dynamically.  
3. **Retrieves relevant code snippets** based on user queries.  
4. **Generates AI-powered insights** using GPT-4 with real-time Markdown rendering.  

---

## **🛠 TODO**
- [ ] Add `.ignore` file support to skip specific directories/files during encoding.  
- [ ] Write a configuration script for setting `OPENAI_API_KEY`.  
- [ ] Add a **web-based interface** for a more interactive experience.  

---

## **🐝 License**
MIT License

