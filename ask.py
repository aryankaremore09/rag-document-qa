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
    prompt = f"""You are answering questions using only the provided context.
If the answer isn't in the context, say you don't know — don't guess.

Context: {context}

Question: {question}
"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user",
                   "content":prompt}]
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    print("Ask questions about your documents (type 'quit' to exit)")
    while True:
        question = input("\nYour question: ")
        if question.lower() == "quit":
            break
        answer = askQuestion(question)
        print(f"\nAnswer: {answer}")