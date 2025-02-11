from pathlib import Path

from config import Config
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from rag_engine import RAGEngine

async def main():
    config = Config()
    # logging.basicConfig(level=logging.INFO)

    processor = DocumentProcessor(config)
    docs = await processor.process_document(Path('data/metallica.txt'))

    vector_store = VectorStoreManager(config)
    vector_store.initialize_store(docs)


    rag_engine = RAGEngine(config)
    rag_engine.initialize(vector_store)


    query = 'Who replaced Cliff Burton after his death?'
    result = await rag_engine.query(query)

    print(f"Answer: {result['answer']}")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

