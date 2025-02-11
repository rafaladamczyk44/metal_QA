from dataclasses import dataclass

@dataclass
class Config:
    chunk_size: int = 500
    chunk_overlap: int = 50
    model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2'
    embedding: str = 'all-MiniLM-L6-v2'
    temperature: float = 0.1
    max_tokens: int = 512
    k_documents: int = 3