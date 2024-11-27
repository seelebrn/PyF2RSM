import os
from langchain_chroma import Chroma

def create_or_load_vector_store(doc_objects, embedding_model, folder_name, query=None):
    """Creates or loads a Chroma vector store for a specific folder."""
    vector_store_path = f"./chroma_persist_{folder_name}"
    if os.path.exists(vector_store_path):
        print(f"Loading existing vector store for folder: {folder_name}")
        print(hasattr(embedding_model, 'embed_query'))  # Should return True

        query_embedding = embedding_model.embed_query(query)
        print(f"Query embedding dimensions: {len(query_embedding)}")
        return Chroma(
            persist_directory=vector_store_path,
            embedding_function=embedding_model
        )

    else:
        print(f"Creating a new vector store for folder: {folder_name}")
        vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embedding_model
        )

        print("Adding documents to vector store...")
        texts = [doc.page_content for doc in doc_objects]
        vector_store.add_texts(texts=texts)
        return vector_store
