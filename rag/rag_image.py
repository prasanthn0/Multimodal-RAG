from src.vectordbs.base import VectorDatabaseInterface
from src.config import settings
from src.prompts import PredefinedPrompts
from src.utils import ask_gpt, convert_image_to_base64

class RAGPipelineUsingImage:
    def __init__(self, db_client: VectorDatabaseInterface):
        """
        Initialize the RAG pipeline for field extraction with a retriever and an OpenAI language model.

        Args:
            db_client (ChromaDBRetriever): The vector db instance instance.
        """
        self.database = db_client
        

    def get_query(self, query: str) -> dict:
        # Retrieve relevant documents
        retrieved_docs = self.database.query_vector(query)
        # print(retrieved_docs)
        image_paths = [doc.metadata.get('file_path') for doc in retrieved_docs]
        print(image_paths)

        base64_images = [convert_image_to_base64(image_path) for image_path in image_paths]
        
        prompt = PredefinedPrompts.rag_prompt_template.format(context="", question = query)

        response = ask_gpt(prompt, base64_images)
        
        return (response['response'], response['tokens_used'] , response['prompt_tokens'])

    def process_output(self, response: str) -> dict:
        """
        Process the language model output into a structured dictionary of fields.

        Args:
            response (str): The raw output from the language model.

        Returns:
            dict: A dictionary of extracted fields.
        """
        # This function can parse the response string into key-value pairs. 
        # Assuming response format is something like "Field: Value"
        field_dict = {}
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                field_dict[key.strip()] = value.strip()

        return field_dict
    

if __name__ == "__main__":
    from src.vectordbs.chromadb import ChromaDB
    from src.config import settings
    collection_name = "userimages"
    vector_db = ChromaDB(collection_name=collection_name)
    ans = RAGPipelineUsingImage(vector_db).get_query("what is the length of the contract?")
    print(ans)
