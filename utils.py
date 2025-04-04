import os
import faiss
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

# You can replace 'sentence-transformers/all-MiniLM-L6-v2' with any local or huggingface embedding model you want.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_function():
    """Returns a LangChain-compatible embedding function."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_embedding_dim(embedding_fn):
    """Returns the dimension by embedding a dummy text."""
    test_vector = embedding_fn.embed_query("test")
    return len(test_vector)

def load_faiss_index(vectorstore_path="vectorstore/faiss_index"):
    """Load FAISS index and embeddings from local directory if it exists, else create a new one."""
    embedding_fn = get_embedding_function()
    dimension = get_embedding_dim(embedding_fn)
    pkl_path = os.path.join(vectorstore_path, "faiss_store.pkl")
    if os.path.exists(vectorstore_path) and os.path.exists(pkl_path):
        with open(f"{vectorstore_path}/faiss_store.pkl", "rb") as f:
            faiss_store = pickle.load(f)
        faiss_index = faiss_store.index
        return faiss_store  
    else:
        # Create directory if it doesn't exist
        os.makedirs(vectorstore_path, exist_ok=True)
        # Initialize a new empty vector store
        faiss_index = faiss.IndexFlatL2(dimension)
        # Use InMemoryDocstore to allow add_documents
        docstore = InMemoryDocstore({})
        new_faiss_store = FAISS(
            embedding_function=embedding_fn,
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id={},
        )
    return new_faiss_store

def save_faiss_index(faiss_store, vectorstore_path="vectorstore/faiss_index"):
    """Persist the FAISS index and store to disk."""
    os.makedirs(vectorstore_path, exist_ok=True)
    with open(f"{vectorstore_path}/faiss_store.pkl", "wb") as f:
        pickle.dump(faiss_store, f)

def add_document_to_index(file_path, faiss_store):
    """Process a text file and add embeddings to the FAISS index."""
    # For simplicity, treat the entire file as a single chunk. 
    # For production, you may chunk the text with LangChain text splitters.
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = Document(page_content=text, metadata={"source": os.path.basename(file_path)})

    # Add to vectorstore
    faiss_store.add_documents([doc])

def remove_document_from_index(document_name, faiss_store, data_folder="documents", vectorstore_path="vectorstore/faiss_index"):
    """
    Removes a document by name. This naive approach physically removes the file
    from the data folder (if it exists), then rebuilds the entire FAISS index.
    """
    import os
    import shutil
    
    # 1. Remove or move the file
    file_path = os.path.join(data_folder, document_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed document file: {file_path}")
    else:
        print(f"No document found at: {file_path}")

    # 2. Create a brand-new empty FAISS index
    embedding_fn = get_embedding_function()
    dimension = get_embedding_dim(embedding_fn)
    new_faiss_index = faiss.IndexFlatL2(dimension)
    docstore = InMemoryDocstore({})
    new_faiss_store = FAISS(
        embedding_function = embedding_fn,
        index = new_faiss_index,
        docstore = docstore,  # new docstore
        index_to_docstore_id = {},  # new index_to_docstore_id
    )

    # 3. Re-add all remaining docs in data_folder to the new index
    # Re-add all remaining docs in the data_folder
    remaining_files = os.listdir(data_folder)
    print("Remaining files for re-indexing:", remaining_files)
    for fname in os.listdir(data_folder):
        fpath = os.path.join(data_folder, fname)
        if os.path.isfile(fpath):
            add_document_to_index(fpath, new_faiss_store)
    
    # 4. Save the updated store to disk
    save_faiss_index(new_faiss_store, vectorstore_path=vectorstore_path)

    return new_faiss_store

