"""
This file handles the offline ingestion pipeline where documents are loaded, chunked, embedded using a local embedding model, and stored in a FAISS vector database for efficient semantic retrieval during query time.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os


## load the document
def load_document():
    loader = TextLoader("data/Google.txt", encoding="utf-8")
    documents = loader.load()
    return documents


## split the document into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    chunks = text_splitter.split_documents(documents)
    return chunks


## create embedding and store in FASSI
def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local("vectorstore")


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)

    print("Documents ingested and vector store created!")
