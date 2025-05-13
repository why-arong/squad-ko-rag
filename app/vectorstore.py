# from langchain.vectorstores import Chroma
# TODO: chang db to chroma later.


from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from app.config import index_name

load_dotenv()


class PineconeClinet:

    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key)
        self.vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name,
                                                                   embedding=self.embeddings)

    def get_vectorstore(self):
        return self.vectorstore

    def get_relevant_documents(self, query: str, search_type: str = "similarity_score_threshold", k: int = 1):
        return self.vectorstore.similarity_search(query, search_type=search_type, k=k)

    def get_retrieval(self, search_type: str = "similarity_score_threshold", k: int = 1):
        return self.vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": k, "score_threshold": 0.5})
