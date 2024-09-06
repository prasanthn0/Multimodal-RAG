from src.config import settings
from src.vectordbs.base import VectorDatabaseInterface
from src.image_mode.image_ingestor import ImageDataIngestor
from src.image_mode.pdf2img import convert_pdf_to_images

def load_images(vector_db_client: VectorDatabaseInterface , file_path : str ) -> str:
    """
    Main function to load embeddings into the vector database using OpenAI embeddings.

    Args:
        data_directory (str): Path to the directory containing BidX folders.
        vector_database (str): The name of the vector database (e.g., "chromadb").

    Returns:
        str: A message indicating success or failure.
    """
    
    target_folder = convert_pdf_to_images(file_path)
    
    if 'image' in target_folder.lower():
        ImageDataIngestor(folder_path=target_folder, db_client=vector_db_client).main()
        
    return "Success: All image data ingested successfully."

if __name__ == "__main__":
    vector_database = "chromadb"
    result_message = load_images(
        vector_database=vector_database
    )
    print(result_message)