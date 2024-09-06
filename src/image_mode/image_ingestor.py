from src.vectordbs.base import VectorDatabaseInterface
from typing import Any, Dict, List, Tuple
from src.utils import describe_image, convert_image_to_base64,execute_parallel, clean_text
import os

class ImageDataIngestor:

    def __init__(self, db_client: VectorDatabaseInterface ,folder_path:str , initial_data: str = None  ):
        """
        Initialize the TextDataIngestor with the specified database client and optional initial data.

        Args:
            db_client (Any): The client or instance of the vector database to use (e.g., a MongoDB client, a Weaviate client).
            initial_data (str, optional): Initial text data to ingest and process. Default is None.
            user_id (str, must) : Storing of data under this user_id index. Default is None.
        """
        self.database = db_client
        self.folder = folder_path

    def data_extractor(self, file_path: str) -> Any:
        """
        Extract image data from a PDF file.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            Any: Image data in base64 format.
        """
        image_data = convert_image_to_base64(file_path)
        return image_data

    def transform_data(self, data: Any, file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Transform image data into the required format for ingestion.

        Args:
            data (Any): Extracted image data.
            file_path (str): Path of the file being ingested.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Data dictionary and metadata.
        """
        print("Describing image")
        output = describe_image(data)
        extracted_text = output['extracted_text']
        extracted_text = clean_text(extracted_text)
        description = output['image_description']
        summary = description+" "+extracted_text

        content = {"content":summary}
        metadata = {'file_path': file_path}

        return content, metadata

    def ingest_data(self, file_path: str ) -> None:
        """
        Ingest image data into the ChromaDB database.

        Args:
            image_data (Any): Image data to ingest.
            file_path (str): Path of the file being ingested.
        """
        print("Ingesting image data")
        
        image_data = self.data_extractor(file_path)
        content, metadata = self.transform_data(image_data, file_path)

        self.database.store_vector(content , metadata)

    def main(self) -> None:
        """
        Execute the ingestion pipeline for image data.

        Args:
            file_path (str): Path to the PDF file.
        """
        print("Extracting image data")
        file_paths = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if os.path.isfile(os.path.join(self.folder, f))]
        
        if len(file_paths) > 0:
            execute_parallel(self.ingest_data, file_paths)
