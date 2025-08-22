# 📄 Gemini RAG Document Q&A App

This project is a **Streamlit-based RAG (Retrieval-Augmented Generation) application** that allows you to upload **PDF** or **DOCX** files and ask natural language questions about their content.  
It extracts text, builds embeddings, and uses **Google Gemini** to provide context-aware answers.  

---

## ✨ Features
- Upload **PDF** or **DOCX** documents.
- Automatic **OCR (Optical Character Recognition)** for scanned PDFs using `pytesseract`.
- Intelligent text splitting and vector storage with **FAISS**.
- Semantic embeddings using **HuggingFace BGE-small**.
- Question answering powered by **Google Gemini (1.5-flash)** for fast and efficient responses.
- Streamlit-based interactive UI.

---

## ⚡ Tech Stack
- Python 3.9+
- Streamlit – UI framework
- LangChain – Orchestration
- FAISS – Vector database
- HuggingFace BGE – Embeddings
- Google Gemini – LLM for Q&A
- pytesseract + pdf2image – OCR for scanned PDFs

---


## Screenshorts


<img width="975" height="589" alt="image" src="https://github.com/user-attachments/assets/73a6d197-df92-4a7b-ba8b-6f7c72aefd69" />



