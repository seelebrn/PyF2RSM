from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def build_retrieval_chain(retriever, answering_model):
    """Builds the retrieval chain."""
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template="""
You are a mental health researcher analyzing posts from an online community about mental health concerns. 
Your role is to extract insights based on the posts provided and deliver responses grounded only in those posts, avoiding assumptions or generic information. 
Respond in a professional and concise manner, focusing on trends and themes in the data.

Context: {context}

Question: {input}
Response:
"""
    )
    combine_docs_chain = create_stuff_documents_chain(
        llm=answering_model,
        prompt=prompt
    )
    return create_retrieval_chain(retriever, combine_docs_chain)
