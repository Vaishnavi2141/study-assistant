import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Global chat history for session memory
chat_history = []

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="./vectorstore",
        embedding_function=embeddings
    )
    return vectorstore

def create_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    llm = ChatGoogleGenerativeAI(
       model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful study assistant for students.
Use the following context from the student's study material to answer the question.
Always be clear, accurate and educational in your response.
If the answer is not in the context, say 'I couldn't find this in your study material.'

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def generate_practice_questions(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke("main topics and concepts")
    context = "\n".join([doc.page_content for doc in docs])
    llm = ChatGoogleGenerativeAI(
      model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )
    question_prompt = f"""You are an expert teacher. Based on the following study material, 
generate 5 exam-style practice questions that test deep understanding.
Format them as a numbered list.
Make them specific to the content provided.

Study Material:
{context}

Generate 5 practice questions:"""

    response = llm.invoke(question_prompt)
    return response.content

def get_answer(chain, retriever, question):
    # Get relevant docs for citations
    docs = retriever.invoke(question)
    citations = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        citations.append(f"Page {page + 1}")
    citations = list(set(citations))

    # Get answer
    answer = chain.invoke(question)

    # Update chat history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer, citations

if __name__ == "__main__":
    print("Loading RAG chain...")
    chain, retriever = create_rag_chain()
    print("RAG chain ready!")

    question = "What is the main theme of The Last Lesson?"
    print(f"\nQuestion: {question}")
    answer, citations = get_answer(chain, retriever, question)
    print(f"\nAnswer: {answer}")
    print(f"\nSources: {', '.join(citations)}")

    print("\n--- Practice Questions ---")
    vectorstore = load_vectorstore()
    questions = generate_practice_questions(vectorstore)
    print(questions)