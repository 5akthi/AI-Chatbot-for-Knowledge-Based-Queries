import os
import pinecone
import langchain
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_ENV'] = PINECONE_ENV
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

def doc_preprocessing(path):
    loader = DirectoryLoader(
        path,
        glob = '**/*.pdf',
        show_progress=True,
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    return docs

def split_docs(docs):
  text_splitter = CharacterTextSplitter(
     separator="\n",
     chunk_size=1000,
     chunk_overlap=20,
     length_function=len
     )
  docs = text_splitter.split_documents(docs)
  return docs

llm = OpenAI()
embeddings = OpenAIEmbeddings()

@st.cache_resource
def store_embeddings(path):
    documents = doc_preprocessing(path)
    docs = split_docs(documents)
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    index = Pinecone.from_documents(
        docs,
        embeddings,
        index_name='demo-index'
    )
    
@st.cache_resource
def retrieve_answer(query):
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    index_name = "demo-index"
    embeddings_db = Pinecone.from_existing_index(
        index_name,
        embeddings
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever = embeddings_db.as_retriever()
    )
    query = query
    result = qa.run(query)
    return result

def main():
    st.set_page_config(page_title="AI Chatbot")
    st.header("AI Chatbot for Knowledge base Queries")
    # store_embeddings('data/')

    # show user input
    user_question = st.text_input("Ask your Query:")
    if st.button("Ask Query"):
        if len(user_question) > 0:
            response = retrieve_answer(user_question)
            st.info(response)
            
if __name__ == "__main__":
    main()