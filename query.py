from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate


VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"
TOP_K = 3
DEBUG = False  # Set True only when debugging retrieval


def load_vector_store():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore


def retrieve_documents(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(query)
    if DEBUG:
        print("\n--- Retrieved Context ---")
        for i, doc in enumerate(docs):
            print(f"\nChunk {i + 1}:")
            print(doc.page_content)

    return docs


def generate_answer(docs, query):
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PromptTemplate(
        template="""
        You are an assistant that answers questions using ONLY the provided context.
        If the answer is not explicitly present in the context, say:
        "The document does not provide this information."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"],
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    final_prompt = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content


if __name__ == "__main__":
    vectorstore = load_vector_store()
    user_question = input("Ask your question: ").strip()
    if not user_question:
        print("Please enter a valid question.")
        exit(1)

    docs = retrieve_documents(vectorstore, user_question)
    answer = generate_answer(docs, user_question)

    print("\n Answer:")
    print(answer)
