import os
from pydantic_settings import BaseSettings
from typing import Any

class Settings(BaseSettings):
    PROJECT_PATH: str  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    SCRIPT_PATH: str = os.path.dirname(os.path.realpath(__file__))
    DATA_STORAGE: str = os.path.join(PROJECT_PATH, "data")
    IMAGES_STORAGE: str = "images"

    #Databases
    VECTOR_DB: str = "chromadb"
    DB_NAME: str = "./chroma"
    SUPPORTED_DATABASES :list = ['chromadb']
    TEXT_COLLECTION_NAME:str = "usertext"
    IMAGE_COLLECTION_NAME:str = "userimages"

    # OpenAI
    CHAT_MODEL: str = "gpt-4o-mini"
    EMBEDDINGS_MODEL: str = "text-embedding-ada-002"
    TEMPERATURE: float = 0.8
    OPENAI_API_KEY: str = "INSERT YOUR API KEY HERE"  
     
    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    CHUNK_TYPE: str = "semantic"
    MAX_TOKEN_LIMIT:int = 10000

    # Retrieval
    CHAIN_TYPE: str = "stuff"
    NUM_DOCS: int = 5
    

settings = Settings()
