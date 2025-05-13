from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    retrieved_document_id: str
    retrieved_document: str
    question: str
    answer: str