from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from config import EMBEDDING_MODEL_NAME, ANSWERING_MODEL_NAME, ANSWERING_MODEL_URL

def get_embedding_model():
    """Returns the embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_answering_model():
    """Returns the answering model."""
    return Ollama(model=ANSWERING_MODEL_NAME, base_url=ANSWERING_MODEL_URL)
