from crewai_tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

PDF_PATH = "./Guidelines/EMS98_Original_english__earthquake.pdf"
CHROMA_DB_DIR = "./offline_chroma_db"

@tool
def offline_pdf_search_tool(query: str) -> str:
    """Offline semantic search over a local earthquake guidelines PDF."""

    # Use huggingface embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2"
    )
    embedding_model.client.max_seq_length = 512

    # If DB exists, load it
    if os.path.exists(CHROMA_DB_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding_model
        )
    else:
        # Else, load and index the PDF
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=CHROMA_DB_DIR
        )
        vectorstore.persist()

    # Run search
    results = vectorstore.similarity_search(query, k=3)

    # Format output
    return "\n\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(results)])
