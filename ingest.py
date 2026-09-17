import os
from pypdf import PdfReader
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = OpenAI()

def extractText(filepath):
    reader = PdfReader(filepath)
    print(reader)
    text = ""
    for page in reader.pages:
        text+= page.extract_text() + "\n"
        # print("--------------------------------------------------------------------")
        # print(text)
        # print("--------------------------------------------------------------------")
    return text

def chunkText(text, chunk_size = 2000, overlap = 200):
    # words = text.split()
    # chunks = []
    # i = 0
    # while i < len(words):
    #     chunk = " ".join(words[i:i + chunk_size])
    #     chunks.append(chunk)
    #     i+=chunk_size - overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = overlap,
        separators=["\n\n","\n",". "," ",""]
    )

    return splitter.split_text(text)

chromaClient = chromadb.PersistentClient(path="./chroma_db")
collection = chromaClient.get_or_create_collection(name="my_documents")

docs_folder = "docs"
chunkId = 0

for fileName in os.listdir(docs_folder):
    if fileName.endswith(".pdf"):
        print(f"Processing {fileName}......")
        filepath = os.path.join(docs_folder, fileName)
        text = extractText(filepath)
        chunks = chunkText(text)

        for chunk in chunks:
            response = client.embeddings.create(
                model = "text-embedding-3-small",
                input=chunk
            )

            embedding = response.data[0].embedding

            collection.add(
                ids = [str(chunkId)],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": fileName}]
            )

            chunkId+=1

print(f"Done. Stored {chunkId} chunks from your documents.")