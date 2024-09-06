from typing import List
from src.text_mode.ingestor import TextDataIngestor, TableDataIngestor ,ImageDataIngestor
from src.config import settings
from src.vectordbs.base import VectorDatabaseInterface

def load_embeddings(vector_db_client: VectorDatabaseInterface, file_path : str) -> str:
    """
    Main function to load embeddings into the vector database using OpenAI embeddings.

    Args:
        data_directory (str): Path to the directory containing BidX folders.
        vector_database (str): The name of the vector database (e.g., "chromadb").

    Returns:
        str: A message indicating success or failure.
    """
    # Ingest data into ChromaDB based on file format
    if file_path.endswith(('.pdf', '.docx', '.html')):
        TextDataIngestor(db_client=vector_db_client).main(file_path)
    if file_path.endswith(('.pdf', '.docx')):
        TableDataIngestor(db_client=vector_db_client).main(file_path)
        ImageDataIngestor(db_client=vector_db_client).main(file_path)
            
    return "Success: All data ingested successfully in text format."

