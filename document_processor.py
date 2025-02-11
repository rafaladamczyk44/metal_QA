from pathlib import Path
from typing import List
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config

class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            # Bit more fancy splitters for more natural chunks
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

    async def process_document(self, document_path: Path) -> List[Document]:
        """
        Load and split documents into chunks
        :param document_path: path to the file(s)
        :return: List of Document instances (chunked data)
        """
        try:
            loader = TextLoader(str(document_path))
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logging.error('Error Processing document')
            raise