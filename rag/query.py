from rag.rag_image import RAGPipelineUsingImage
from rag.rag_text import RAGPipelineUsingText
from src.vectordbs.chromadb import ChromaDB
from src.config import settings

if __name__ == "__main__":
    query = "what is the Title of the document"
    collection_name = settings.IMAGE_COLLECTION_NAME
    vector_db_client = ChromaDB(collection_name=collection_name)
    ans = RAGPipelineUsingImage(vector_db_client).get_query(query)
    print(ans)

    collection_name = settings.TEXT_COLLECTION_NAME
    vector_db_client = ChromaDB(collection_name=collection_name)
    ans =ans = RAGPipelineUsingText(vector_db_client).get_query(query)
    print(ans)

