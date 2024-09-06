from chromadb import PersistentClient
from langchain_chroma import Chroma
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from typing import Any, Dict, List, Tuple, Optional
from src.vectordbs.base import VectorDatabaseInterface
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.config import settings
from langchain_core.documents import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


class ChromaDB(VectorDatabaseInterface):
    def __init__(self, collection_name: str):
        """
        Initialize ChromaDB client and create a collection.

        Args:
            collection_name (str): Name of the ChromaDB collection to use.
        """
        self.client = PersistentClient(
                        path=settings.DB_NAME,
                        settings=Settings(),
                        tenant=DEFAULT_TENANT,
                        database=DEFAULT_DATABASE,
                    )
        self.collection_name = collection_name
        self.client.get_or_create_collection(name=collection_name)
        self.embeddings_model = OpenAIEmbeddings(openai_api_key = settings.OPENAI_API_KEY)
        self.LLM = ChatOpenAI(temperature=settings.TEMPERATURE, model=settings.CHAT_MODEL, openai_api_key = settings.OPENAI_API_KEY)

        self.vector_store = Chroma(
                                    client=self.client,
                                    collection_name=self.collection_name,
                                    embedding_function=self.embeddings_model,
                                )
        
        document_content_description = "contents and summary of a pdf page"
        metadata_field_info = [
                                AttributeInfo(
                                    name="image_path",
                                    description="Path to where the base image is stored",
                                    type="string",
                                )
                                ]
        self.retriever = SelfQueryRetriever.from_llm(
                        self.LLM, self.vector_store, document_content_description, metadata_field_info, verbose=True,k=settings.NUM_DOCS
                    )
        
    def reset_database(self):
        self.client.delete_collection(name=self.collection_name)
        self.client.get_or_create_collection(name=self.collection_name)

    def compute_embedding(self, text: str) -> List[float]:
        """
        Compute embeddings for the provided text using OpenAIEmbeddings from Langchain.

        Args:
            text (str): The input text to compute the embedding.

        Returns:
            List[float]: The computed embedding as a list of floats.
        """
        # Generate embeddings using OpenAIEmbeddings
        return self.embeddings_model.embed_documents(text)

    def store_vector(self, data: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> str:
        """
        Store the vector and metadata in the ChromaDB collection.

        Args:
            data (Dict[str, Any]): Contains {"content": <original_text>, "transformed_content": <optional altered text>}.
            metadata (Dict[str, Any]): Additional metadata to store with the vector.

        Returns:
            str: Success message upon successful insertion.
        """
        print("Inserting a document into ChromaDB")
        
        # Extract content and transformed content
        content = data["content"]
    
        doc = Document(
                        page_content=content,
                        metadata = metadata,
                    )
        self.vector_store.add_documents(documents=[doc])
        return "Success"

    def query_vector(self, query: str, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the ChromaDB collection and return matching vectors for a given query string.

        Args:
            query (str): The query string to search for in the vector database.
            top_k (int): Number of results to return. Default is 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing matched vectors' content and metadata.
        """
        docs = self.retriever.invoke(query)
        return docs

    def set_memory(self, history: List[Dict[str, Any]]) -> str:
        """
        Store chat history in the ChromaDB collection.

        Args:
            history (List[Dict[str, Any]]): A list of chat messages or history items to store.

        Returns:
            str: Success message upon successful insertion.
        """
        # Iterate over chat history and store each message as a separate document in ChromaDB
        for item in history:
            self.store_vector(data={"content": item['message'], "transformed_content": item.get('embedding')}, 
                              metadata={"timestamp": item['timestamp']})
        return "Memory set successfully"
