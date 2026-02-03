# Local RAG Application (LangChain + Ollama + FAISS)

This project is a **local Retrieval-Augmented Generation (RAG)** application built using **LangChain**, **Ollama**, and **FAISS**.  
It allows users to ask questions over their own documents and receive **grounded, non-hallucinated answers**.

---

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique where:

1. Relevant documents are retrieved using embeddings
2. Retrieved content is provided as context to a language model
3. The LLM generates answers grounded strictly in the retrieved data

This improves **accuracy**, **trust**, and **explainability** compared to standalone LLMs.

---

## Architecture Overview

Documents (TXT / PDF)
↓
Text Chunking
↓
Embeddings (Ollama)
↓
Vector Store (FAISS)
↓
Similarity Search
↓
Context + Question
↓
LLM Answer (ChatOllama)

---

## Setup Instructions

### Create Virtual Environment

```bash
python -m venv venv

venv\Scripts\activate

## Install Dependencies

pip install langchain langchain-community langchain-text-splitters langchain-ollama faiss-cpu

Install Required Ollama Models

ollama pull llama3.2:1b
ollama pull nomic-embed-text

<!-- How to Run -->

python ingest.py

Ask Questions (Runtime)

python query.py


```
