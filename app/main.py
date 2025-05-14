from fastapi import FastAPI, HTTPException

from app.llms.huggingface import HuggingFaceLLM
from app.llms.upstage import UpstageLLM
from app.models import QueryRequest, QueryResponse
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: init db
    yield


app = FastAPI(lifespan=lifespan)

# TODO: make base LLM!! (use abc meta class later...)
# llm = UpstageLLM()
llm = HuggingFaceLLM()


@app.get("/")
def read_root():
    return {"message": "Ask me everything about Wikipedia!!"}


@app.post("/query", response_model=QueryResponse)
async def rag_endpoint(query: QueryRequest):
    try:
        return llm.generate_answer(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
