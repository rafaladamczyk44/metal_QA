from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config


class VectorStoreManager:
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding)
        self.vector_store = None

    def initialize_store(self, documents: List[Document]):
        # Similarity search
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
