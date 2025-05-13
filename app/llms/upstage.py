from langchain_upstage import ChatUpstage
from app.vectorstore import PineconeClinet
from app.models import QueryRequest, QueryResponse
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


class UpstageLLM:
    def __init__(self):
        self.model = ChatUpstage()

    def generate_answer(self, query: QueryRequest):
        user_question = query.question
        vectorstore = PineconeClinet()
        retrieval = vectorstore.get_retrieval()

        document_list = retrieval.get_relevant_documents(user_question)

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        combine_docs_chain = create_stuff_documents_chain(self.model, retrieval_qa_chat_prompt)
        rag_chain = create_retrieval_chain(retrieval, combine_docs_chain)

        metadata = document_list[0].metadata
        retrieved_document_id = metadata["id"]
        retrieved_document = document_list[0].page_content
        answer = rag_chain.invoke({"input": user_question})['answer']
        return QueryResponse(
            question=user_question,
            answer=answer,
            retrieved_document_id=retrieved_document_id,
            retrieved_document=retrieved_document
        )
