import streamlit as st
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

chromaClient = chromadb.PersistentClient(path="./chroma_db")
collection = chromaClient.get_or_create_collection(name="my_documents")

def askQuestion(question):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    questionEmbedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[questionEmbedding],
        n_results=4
    )

    retrievedChunks = results["documents"][0]

    context = "\n\n".join(retrievedChunks)
    prompt = f""" Always answer question based on the context. If you don't know the answer just say i don't know. Don't Hallucinate.
Context: {context}
Question: {question}
"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages= [{"role":"user", "content":prompt}] 
    )

    return completion.choices[0].message.content


st.title("QnA bot - Ask questions:")
st.write("Ask questions based on documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask your question.......")

if question:
    st.session_state.messages.append({"role":"user", "content":question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = askQuestion(question)
        st.write(answer)

    st.session_state.messages.append({"role":"user","content":answer})




    

