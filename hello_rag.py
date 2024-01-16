#!/usr/bin/env python

# Retrieves documents from a PDF file, splits them into smaller documents,
# encodes them into embeddings, indexes them in a vector store, and creates
# a retrieval chain to find relevant documents for a given question.

import os
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from utils import simulate_typing


load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")


# Import necessary libraries and classes
llm = ChatOpenAI(
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ],  # Callback to handle streaming standard output
    openai_api_key=open_api_key,  # Set the OpenAI API key
)

# Create a prompt template for asking questions based on context
prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)

# Load a PDF file and split it into documents
loader = PyMuPDFLoader("./llm-ebook-part1.pdf")
docs = loader.load()

# Use a text splitter to separate characters in the documents
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Create embeddings using OpenAI's model
embeddings = OpenAIEmbeddings()

# Generate a vector using FAISS from the documents and embeddings
vector = FAISS.from_documents(documents, embeddings)

# Parse the output as a string
output_parser = StrOutputParser()

# Create a document chain for processing documents
document_chain = create_stuff_documents_chain(llm, prompt)

# Use the vector as a retriever
retriever = vector.as_retriever()

# Create a retrieval chain for retrieving relevant information
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Continuously interact with the user
while True:
    # Get user input
    question = input("User: ")

    # Check if the user wants to exit
    if question == "exit":
        break

    # Invoke the retrieval chain to get a response based on the user's question
    response = retrieval_chain.invoke({"input": question})

    # Print the response with a simulated typing effect
    print(simulate_typing(response["answer"]))
