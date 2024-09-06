from src.image_mode.main import load_images
from src.text_mode.main import load_embeddings
from src.config import settings
import os
from src.vectordbs.chromadb import ChromaDB
from concurrent.futures import ThreadPoolExecutor

def ingest_file(file_path):
    vector_database = settings.VECTOR_DB
    data_directory = settings.DATA_STORAGE
    vector_database = vector_database.lower()

    if not os.path.exists(data_directory) or not os.path.isdir(data_directory):
        return "Error: Data directory does not exist."
    
    if not os.path.exists(file_path):
        print("Error: File does not exist.")

    # Check for supported vector databases
    if vector_database not in settings.SUPPORTED_DATABASES:
        return "Error: Vector database not supported."
    
    print("##########################################################################")
    print("Ingesting file using image mode")
    if vector_database == "chromadb":
        collection_name = settings.IMAGE_COLLECTION_NAME  
        vector_db_client = ChromaDB(collection_name=collection_name) 
    result_message = load_images(vector_db_client=vector_db_client, file_path=file_path)
    print(result_message)

    
    print("##########################################################################")
    print("Ingesting file using text mode")
    if vector_database == "chromadb":
        collection_name = settings.TEXT_COLLECTION_NAME  
        vector_db_client = ChromaDB(collection_name=collection_name) 
    result_message = load_embeddings(vector_db_client=vector_db_client, file_path=file_path)
    print(result_message)

    # # Parallelize both ingestion modes
    # with ThreadPoolExecutor() as executor:
    #     executor.submit(ingest_image_mode)
    #     executor.submit(ingest_text_mode)

    return "File ready to use"

if __name__ == "__main__":
    path = r"data/JA-207652.pdf"
    ingest_file(path)
