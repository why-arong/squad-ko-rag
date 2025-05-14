# Korean Wikipedia RAG Pipeline

---

## Overview

A Retrieval‑Augmented Generation (RAG) pipeline
that answers questions in Korean by retrieving relevant passages from a vector search index built on Korean Wikipedia (
KorQuAD 1.0 train split)
and generating natural‑language answers with a large language model.

## Features

- **Vector search** powered by **Pinecone** with 1024‑dimensional dense embeddings
- **Embeddings & LLM** from Hugging Face
    - LLM model: `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`
    - Embedding model: `nlpai-lab/KURE-v1`
- Upstage cloud models for users without a local GPU (invoke via API)
- **FastAPI** HTTP API with a single `/query` endpoint

## Dataset

We use the training portion of **KorQuAD 1.0**—a 70 k+ question‑answer corpus created from Korean Wikipedia articles—for
both knowledge base and evaluation.  
For license and statistics, see the official [KorQuAD site](https://korquad.github.io/).

## Architecture

```mermaid
graph TD
    A[Client] -->|POST /query| B[FastAPI]
    B --> C[Retriever<br/>Pinecone]
    C --> D[Top‑k Passages]
    D --> E[Generator<br/>LLM]
    E --> F[JSON Answer]
   ```

--- 

## Quickstart

- **Configure `.env` file based on [section below](#environment-variables)**
- **Setup Pinecone Vectorstore:**
    - Use notebooks/kor_squad_to_vectorstore.ipynb to store KorQuAD data into Pinecone.
    - The default embedding model outputs 1024-dimensional vectors. Be sure to set the Pinecone index accordingly.

- **Run the API locally**:
    - Ensure all required environment variables are set!
    - Run the following commands (preferably in
      a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/))

```bash
git clone https://github.com/why-arong/squad-ko-rag.git
cd squad-ko-rag
```

```bash
pip install -r requirements.txt
uvicorn app.main:app --host {HOST_ADDRESS} --port 4000
```

### Environment Variables

The following environment variables are required to run the application:

- `PINECONE_API_KEY`: The API key for Pincone.
- `UPSTAGE_API_KEY`: *(Optional)* API key for Upstage cloud inference, useful if you do not have a GPU.

Create a .env file in the project root and define these variables.

## Usage

Send a POST request to the API endpoint to receive an LLM-generated response.
Questions must relate to the KorQuAD dataset. Irrelevant questions will receive a fallback message:
"제가 가지고 있는 지식으로는 대답할 수 없습니다."

### Example Queries

```shell
curl -X 'POST' \
  '{BASE URL}/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "비트겐슈타인이 중위로 복무할 당시 그에게 데이비드 핀센트에 관한 편지를 보낸 이는?"
}'
```

```shell
curl -X 'POST' \
  '{BASE URL}/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "너 내가 누군지 아니?"
}'
```

### Roadmap

- To be added.