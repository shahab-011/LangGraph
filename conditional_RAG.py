import os
from typing import TypedDict , Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

#  Step 1 - Building the RAG retriever

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)


def build_retriever(pdf_path : str) :
    loader = PyMuPDFLoader(pdf_path)
    document = loader.load

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100
    )

    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks,embeddings)

    return vectorstore.as_retriever(search_kwargs = {"k":4})


acedemic_retriever = build_retriver("academics_handbook.pdf")
fee_retriever = build_retriver("fee_structure.pdf")

# 2 - LLM Model
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4
)


#  Step 3 - State 
class State(TypedDict):
    programme : str
    messages : Annotated[list,add_messages]
    query_type : str
    retrieved_context : str



# Step 4 - Creating Nodes  
def classifier_node(state : State)-> dict:
    #doc string
    """Look at the latest user message and decide which path to take"""     
    last_messgae = state['messages'][-1].content