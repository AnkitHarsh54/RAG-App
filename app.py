import os
import streamlit as st
from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from docx import Document
from tempfile import NamedTemporaryFile

# 🔑 Pulls Gemini API key from Streamlit
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

def ocr_pdf(file_path):
    """Run OCR on scanned PDFs to recover text."""
    images = convert_from_path(file_path)
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text.strip() if text.strip() else None

def load_document(file_path):
    """Read text from PDF/DOCX files. Falls back to OCR if PDF is image-based."""
    if file_path.endswith(".pdf"):
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text = "\n".join([page.page_content for page in pages])
            if not text.strip():
                text = ocr_pdf(file_path)  # fallback if no extractable text
            return text if text else None
        except Exception:
            return None

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    return None

def process_text(text):
    """Breaks text into chunks and builds FAISS index with embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len, is_separator_regex=False
    )
    chunks = splitter.create_documents([text])
    
    model_name = "BAAI/bge-small-en"
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)
    
    chunk_texts = [chunk.page_content for chunk in chunks]
    db = FAISS.from_texts(chunk_texts, embeddings)
    return db

def generate_response(db, query):
    """Search relevant text chunks and use Gemini to answer the query."""
    results = db.similarity_search(query, k=5)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based only on the following context: {context}"),
        ("human", "{question}"),
    ])
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
    chain = prompt | model
    
    response = chain.invoke({
        "context": "\n\n".join([r.page_content for r in results]),
        "question": query
    })
    return response.content

# ------------------ Streamlit UI ------------------

st.title("📄 Document Q&A with Gemini RAG")

uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
query = st.text_input("Ask something about the document:")

if uploaded_file and query:
    with NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.name.endswith(".pdf") else ".docx") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    
    doc_text = load_document(temp_path)
    
    if doc_text:
        db = process_text(doc_text)
        answer = generate_response(db, query)
        st.subheader("Gemini’s Answer:")
        st.write(answer)
    else:
        st.error("⚠️ Couldn’t extract text. The file may be scanned or unsupported.")
