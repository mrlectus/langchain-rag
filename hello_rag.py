#!/usr/bin/env python
import os
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils import simulate_typing


load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=open_api_key)

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)

loader = PyPDFLoader("./llm-ebook-part1.pdf")
docs = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

vector = FAISS.from_documents(documents, embeddings)
output_parser = StrOutputParser()
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

while True:
    question = input("User: ")
    if question == "exit":
        break
    response = retrieval_chain.invoke({"input": question})
    print(simulate_typing(response["answer"]))
