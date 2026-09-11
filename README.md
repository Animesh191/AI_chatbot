This model is based on RAG Pipelines like embedding the documents and then create chunks of the data and then align it in the vector db like pinecone or chrOMADB and then process that.

## Run locally

Use Python 3.11 or newer, install the dependencies from the nested `Pipfile`, and make sure Ollama is running with the `qwen2.5:3b` model:

```powershell
cd AI_AGENT_DOCUMENTATION
python -m pip install certifi langchain langchain-chroma langchain-classic langchain-ollama langchain-pinecone langchain-tavily python-dotenv streamlit
ollama pull qwen2.5:3b
streamlit run main.py
```

The app uses the checked-in `chroma_db` database and opens at `http://localhost:8501`.

`TAVILY_API_KEY` is required only when running `ingestion.py` to refresh the documentation database. Pinecone credentials are not required for local Chroma serving.
