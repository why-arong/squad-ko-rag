from fastapi import FastAPI, HTTPException

from app.llms.upstage import UpstageLLM
from app.models import QueryRequest, QueryResponse
from app.rag_pipeline import generate_answer
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: init db
    yield


app = FastAPI(lifespan=lifespan)

llm = UpstageLLM()


@app.get("/")
def read_root():
    return {"message": "Ask me everything about Wikipedia!!"}


@app.post("/query", response_model=QueryResponse)
async def rag_endpoint(query: QueryRequest):
    try:
        return llm.generate_answer(query)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
