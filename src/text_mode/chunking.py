from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from src.config import settings
from langchain_openai import OpenAIEmbeddings


class ChunkingStrategy:
    def __init__(self, documents):
        self.chunk_type = settings.CHUNK_TYPE
        self.documents = documents[0]
        self.embeddings = OpenAIEmbeddings(openai_api_key = settings.OPENAI_API_KEY)
        
    def split_texts(self):
        if self.chunk_type == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(self.documents)
            return texts
        
        elif self.chunk_type == "semantic":
            texts = [doc.page_content for doc in self.documents if doc.page_content]
            text_splitter = SemanticChunker(self.embeddings)
            processed_texts = text_splitter.create_documents(texts)
            return processed_texts
        
        else:
            raise NotImplementedError("Chunking method not implemented")
