#  rag_build.py
#  Build step: load a resume PDF -> chunk it -> embed it -> create a FAISS vector store.
#  Same pipeline as the class rag_build notebook, wrapped in a function so app.py
#  can call it on resumes uploaded at runtime.

#1. Import all the libraries

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

#2. Load the API keys (fallback if no key is typed in the sidebar)

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"


def build_vectorstore(pdf_path, api_key):
    """Build and return a FAISS vector store from a single resume PDF."""

    #3. Load the PDF file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    #4. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    #5. Create Embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)

    #6. Create vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore