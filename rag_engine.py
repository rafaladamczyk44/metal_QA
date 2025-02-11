import os
from typing import Dict, Any
import logging

from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

from config import Config
from vector_store_manager import VectorStoreManager


class RAGEngine:
    def __init__(self, config: Config):
        self.config = config
        # HF API
        self.llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
            repo_id=config.model_name,
            task='text-generation',
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        self.vsm = None
        self.qa_chain = None

    def initialize(self, vsm: VectorStoreManager):
        # Initialize vector store
        self.vsm = vsm
        retriever = vsm.vector_store.as_retriever(
            search_kwargs={'k': self.config.k_documents}
        )

        # Initialize chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=retriever,
        )

    async def query(self, question:str) -> Dict[str, Any]:
        try:
            # Get the result from FAISS and generate answer to the query
            response = self.qa_chain.invoke({'query': question})
            return {
                'answer': response['result'],
                'source_documents': response.get('source_documents', []),
            }
        except Exception as e:
            logging.error('Error processing the query')
            raise
