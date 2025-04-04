from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import os
import shutil
import requests

from utils import (
    load_faiss_index,
    save_faiss_index,
    add_document_to_index,
    remove_document_from_index,
)

app = FastAPI()

# Load or create FAISS index at startup
faiss_store = load_faiss_index()
DOCS_FOLDER = "documents"

# Mistral API details
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "94DfmNYxZqPVRZct7Ksjao7Tj2lvjA8P"  # Or use an environment variable


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Endpoint to upload a plain text file and index it."""
    if file.content_type != "text/plain":
        raise HTTPException(
            status_code=400,
            detail="Only text/plain files are supported for simplicity."
        )

    # Save file locally
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    file_path = os.path.join(DOCS_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Add to FAISS index
    add_document_to_index(file_path, faiss_store)
    # Persist updated index
    save_faiss_index(faiss_store)

    return {"message": f"File '{file.filename}' uploaded and indexed successfully."}


@app.delete("/delete/{filename}")
def delete_document(filename: str):
    """Endpoint to delete a document file and remove it from the index (naively)."""
    file_path = os.path.join(DOCS_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found."
        )

    # Remove the file
    os.remove(file_path)

    # Remove or rebuild the index as needed
    global faiss_store
    faiss_store  = remove_document_from_index(filename, faiss_store, data_folder="documents")
    save_faiss_index(faiss_store)

    return {"message": f"Document '{filename}' deleted. (Index updated naively)"}


@app.post("/chat")
def chat_with_docs(request: dict):
    """Endpoint to query the vector store + Mistral with minimal payload."""
    query = request.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    # 1) Retrieve relevant chunks from FAISS
    k = 3
    docs = faiss_store.similarity_search(query, k=k)

    # 2) Build a context from the retrieved docs
    context = ""
    for i, d in enumerate(docs):
        context += f"Document {i+1}:\n{d.page_content}\n\n"

    # 3) Construct a final prompt
    final_prompt = (
        f"You are an AI assistant. "
        f"Using the following context:\n\n{context}\n"
        f"Answer the user query:\n{query}"
    )

    # 4) Minimal Mistral payload
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {
                "role": "user",
                "content": final_prompt
            }
        ]
        # If needed, you can add more fields here (temperature, etc.)
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }

    # 5) Send request to Mistral
    try:
        response = requests.post(
            MISTRAL_API_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # 6) Parse response. Adjust to match Mistral's actual schema
        if "choices" in data and len(data["choices"]) > 0:
            answer = data["choices"][0]["message"]["content"]
        else:
            answer = "No answer returned by Mistral."

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with Mistral API: {str(e)}"
        )

    return {"answer": answer}


@app.exception_handler(Exception)
def general_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )
