

## Configure the env 
```bash
conda create -n code_reader python=3.10 -y
conda activate code_reader

pip install langchain openai chromadb tiktoken faiss-cpu
pip install -U langchain-community
pip install langchain_openai
pip install rich
pip install watchdog
```

## Usage

## 1. Configure your open-ai key

```bash 
export OPENAI_API_KEY='your-open-ai-api-key'
```

## 2. Interact with your codebase with command-line tools
```
python3 run.py [your code base root folder]

```

## TODO

- [ ] Add `.ignore` file to skip certain directory / file encoding.
- [ ] Writup program for configure `OPENAI_API_KEY` 
- [ ] Add web brower to make it more enjoyable.
