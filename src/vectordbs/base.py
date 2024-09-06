from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class VectorDatabaseInterface(ABC):
    
    @abstractmethod
    def store_vector(self, data: Dict[str,Any]=None, metadata : Dict[str, Any] = None) -> None:
        """
        Store the vector in the database.

        Args:
            data (Dict[str, Any]): This will contain {"content" :   ,"transformed_content": }
            metadata (Dict[str, Any], optional): Additional metadata to store with the vector. Default is None.
        """
        pass

    @abstractmethod
    def query_vector(self, query: Any, *kwargs) -> List[Any]:
        """
        Query the vector database and return matching vectors.

        Args:
            query (Any): The query to search for in the vector database.

        Returns:
            List[Any]: A list of matching vectors.
        """
        pass

    @abstractmethod
    def set_memory(self, history: List[Dict[str, Any]]) -> Any:
        """
        Set memory or chat history in the vector database.

        Args:
            history (List[Dict[str, Any]]): A list of chat messages or history items to store.

        Returns:
            Any: A LangChain ChatMessageHistory object or equivalent.
        """
        pass

