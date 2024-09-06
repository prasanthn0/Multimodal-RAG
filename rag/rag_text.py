from src.vectordbs.base import VectorDatabaseInterface
from src.config import settings
from src.prompts import PredefinedPrompts
from src.utils import ask_gpt

class RAGPipelineUsingText:
    def __init__(self, db_client: VectorDatabaseInterface):
        """
        Initialize the RAG pipeline for field extraction with a retriever and an OpenAI language model.

        Args:
            db_client (ChromaDBRetriever): The vector db instance instance.
        """
        self.database = db_client
        # self.llm = ChatOpenAI(temperature=settings.TEMPERATURE, model=settings.CHAT_MODEL, openai_api_key = settings.OPENAI_API_KEY)


    def get_query(self, query: str) -> dict:
        # Retrieve relevant documents
        retrieved_docs = self.database.query_vector(query)
        
        # Concatenate the retrieved documents' content to provide context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
        prompt = PredefinedPrompts.rag_prompt_template.format(context=context, question = query)
        
        response = ask_gpt(prompt)
        
        return (response['response'], response['tokens_used'] , response['prompt_tokens'])
    

if __name__ == "__main__":
    from src.vectordbs.chromadb import ChromaDB
    from src.config import settings
    collection_name = "usertext"
    vector_db = ChromaDB(collection_name=collection_name)
    ans = RAGPipelineUsingText(vector_db).get_query("what is the length of the contract?")
    print(ans)
