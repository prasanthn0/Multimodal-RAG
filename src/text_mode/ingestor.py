from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from langchain_core.documents import Document
from src.vectordbs.base import VectorDatabaseInterface
from src.vectordbs.chromadb import ChromaDB  
import tabula
import pdfplumber
import os
from src.utils import execute_parallel, describe_image, get_summary
from src.text_mode.data_formats.doc_loader import load_single_document
from src.text_mode.chunking import ChunkingStrategy
from src.text_mode.data_formats.image_data_extractor import extract_images_and_text_from_pdf

class DataIngestorBase(ABC):
    def __init__(self, db_client: VectorDatabaseInterface):
        """
        Initialize the base data ingestor class with a specified database client.

        Args:
            db_client (VectorDatabaseInterface): The vector database client to use for data storage.
        """
        if db_client is None:
            self.database = ChromaDB(
                collection_name="temp" 
            )
        else:
            self.database = db_client

    @abstractmethod
    def data_extractor(self, file_path: str) -> Any:
        """
        Abstract method to extract data from a file.

        Args:
            file_path (str): Path to the file to be ingested.
        """
        if file_path is None:
            raise Exception("File path not found for data ingestion operation")

    @abstractmethod
    def transform_data(self, data: Any, file_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Abstract method to transform data into the required format for ingestion.

        Args:
            data (Any): Raw data extracted from the file.
            file_name (str): Name of the file being ingested.
        """
        pass

    def summariser(self, data: Any) -> str:
        """
        Summarize the extracted data using a predefined prompt.

        Args:
            data (Any): The extracted data to summarize.

        Returns:
            str: The summary of the extracted data.
        """
        response = get_summary(data)
        return response

    def store_vector(self, data: Dict[str, Any], metadata: Dict[str, Any] = None) -> None:
        """
        Store the vector and metadata in the ChromaDB database.

        Args:
            data (Dict[str, Any]): Data dictionary containing content and transformed content.
            metadata (Dict[str, Any], optional): Additional metadata associated with the vector.
        """
        self.database.store_vector(data, metadata)

    @abstractmethod
    def ingest_data(self, data: Any, file_name: str) -> None:
        """
        Abstract method to ingest data into the database.

        Args:
            data (Any): Data to ingest.
            file_name (str): Name of the file being ingested.
        """
        pass

    @abstractmethod
    def main(self, file_path: str) -> None:
        """
        Abstract method to execute the ingestion pipeline.

        Args:
            file_path (str): Path to the file to be ingested.
        """
        pass

class TextDataIngestor(DataIngestorBase):
    def __init__(self, db_client, initial_data: str = None  ):
        """
        Initialize the TextDataIngestor with the specified database client and optional initial data.

        Args:
            db_client (Any): The client or instance of the vector database to use (e.g., a MongoDB client, a Weaviate client).
            initial_data (str, optional): Initial text data to ingest and process. Default is None.
            user_id (str, must) : Storing of data under this user_id index. Default is None.
        """
        super().__init__(db_client)

    def data_extractor(self, file_path: str) -> List[Document]:
        """
        Extract text data from a document file.

        Args:
            file_path (str): Path to the document file.

        Returns:
            List[Document]: List of extracted document texts.
        """
        super().data_extractor(file_path)
        docs = [load_single_document(file_path)]
        texts = ChunkingStrategy(docs).split_texts()
        return texts

    def transform_data(self, data: Dict[str, Any], file_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Transform text data into the required format for ingestion.

        Args:
            data (Dict[str, Any]): Extracted text data.
            file_name (str): Name of the file being ingested.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Data dictionary and metadata.
        """
        data_dict = {
            'content': str(data['page_content']),
            'transformed_content': data['page_content']
        }
        metadata = {'file_name': file_name}
        return data_dict, metadata

    def ingest_data(self, data: Any, file_name: str) -> None:
        """
        Ingest text data into the ChromaDB database.

        Args:
            data (Any): Text data to ingest.
            file_name (str): Name of the file being ingested.
        """
        print("Ingesting text data")
        data, metadata = self.transform_data(data, file_name)
        self.store_vector(data, metadata)

    def main(self, file_path: str) -> None:
        """
        Execute the ingestion pipeline for text data.

        Args:
            file_path (str): Path to the document file.
        """
        docs = self.data_extractor(file_path)
        file_name = os.path.basename(file_path)
        
        data = [doc.dict() for doc in docs]
        
        if len(docs) > 0:
            execute_parallel(self.ingest_data, data, file_name)

class TableDataIngestor(DataIngestorBase):

    def __init__(self, db_client: Any = None, initial_data: str = None):
        """
        Initialize the TextDataIngestor with the specified database client and optional initial data.

        Args:
            db_client (Any): The client or instance of the vector database to use (e.g., a MongoDB client, a Weaviate client).
            initial_data (str, optional): Initial text data to ingest and process. Default is None.
            user_id (str, must) : Storing of data under this user_id index. Default is None.
        """
        super().__init__(db_client )

    def convert_keys_to_strings(self,dictionary):
        """
        Recursively converts all numeric keys to strings in a dictionary.
        """
        if isinstance(dictionary, dict):
            return {str(key): self.convert_keys_to_strings(value) for key, value in dictionary.items()}
        elif isinstance(dictionary, list):
            return [self.convert_keys_to_strings(item) for item in dictionary if item != "" and item is not None]
        else:
            return dictionary
        
    def data_extractor(self, file_path: str) -> Any:
        """
        Extract table data from a PDF file using Tabula.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            Any: Extracted table data.
        """
        super().data_extractor(file_path)
        try:
            df_tables = tabula.read_pdf(file_path, pages="all")
        except:
            df_tables = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # Extract tables from the current page
                    tables = page.extract_tables()
                    df_tables.extend(tables)
        print(f"{len(df_tables)} tables were extracted from {file_path}")
        return df_tables

    def transform_data(self, data: Any, file_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Transform table data into the required format for ingestion.

        Args:
            data (Any): Extracted table data.
            file_name (str): Name of the file being ingested.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Data dictionary and metadata.
        """
        response = super().summariser(data)
        data_dict = {
            'content': str(self.convert_keys_to_strings(data)),
            'transformed_content': response
        }
        metadata = {'file_name': file_name}
        return data_dict, metadata

    def ingest_data(self, data: Any, file_name: str) -> None:
        """
        Ingest table data into the ChromaDB database.

        Args:
            data (Any): Table data to ingest.
            file_name (str): Name of the file being ingested.
        """
        print("Ingesting table data")
        data, metadata = self.transform_data(data, file_name)
        self.store_vector(data, metadata)

    def main(self, file_path: str) -> None:
        """
        Execute the ingestion pipeline for table data.

        Args:
            file_path (str): Path to the PDF file.
        """
        data = self.data_extractor(file_path)
        file_name = os.path.basename(file_path)
        if len(data) > 0:
            execute_parallel(self.ingest_data, data, file_name)


class ImageDataIngestor(DataIngestorBase):

    def __init__(self, db_client: Any = None, initial_data: str = None  ):
        """
        Initialize the TextDataIngestor with the specified database client and optional initial data.

        Args:
            db_client (Any): The client or instance of the vector database to use (e.g., a MongoDB client, a Weaviate client).
            initial_data (str, optional): Initial text data to ingest and process. Default is None.
            user_id (str, must) : Storing of data under this user_id index. Default is None.
        """
        super().__init__(db_client)

    def data_extractor(self, file_path: str) -> Any:
        """
        Extract image data from a PDF file.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            Any: Extracted image data.
        """
        super().data_extractor(file_path)
        image_data, _ = extract_images_and_text_from_pdf(file_path)
        return image_data

    def transform_data(self, data: Any, file_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Transform image data into the required format for ingestion.

        Args:
            data (Any): Extracted image data.
            file_name (str): Name of the file being ingested.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Data dictionary and metadata.
        """
        try:
            data_dict = {
                'content': str(data),
                'transformed_content': data['extracted_text']+ "\n" + data['image_description']
            }
        except:
            data_dict = {
                'content': str(data),
                'transformed_content': str(data)
            }

        metadata = {'file_name': file_name}
        return data_dict, metadata

    def ingest_data(self, image_data: Any, file_name: str) -> None:
        """
        Ingest image data into the ChromaDB database.

        Args:
            image_data (Any): Image data to ingest.
            file_name (str): Name of the file being ingested.
        """
        print("Ingesting image data")
        image_descriptions = describe_image(image_data)
        data, metadata = self.transform_data(image_descriptions, file_name)
        self.store_vector(data, metadata)

    def main(self, file_path: str) -> None:
        """
        Execute the ingestion pipeline for image data.

        Args:
            file_path (str): Path to the PDF file.
        """
        print("Extracting image data")
        image_data = self.data_extractor(file_path)
        file_name = os.path.basename(file_path)
        if len(image_data) > 0:
            execute_parallel(self.ingest_data, image_data, file_name)
