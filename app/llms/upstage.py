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

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        retrieval_qa_chat_prompt.messages[0].prompt.template += (
            "\n\n당신은 오직 제공된 문서의 내용을 기반으로만 답변해야 합니다. "
            "만약 문서에서 해당 질문에 대한 명확한 정보를 찾을 수 없다면, "
            "'죄송합니다. 제공된 문서에서는 답을 찾을 수 없습니다.'라고 응답하십시오."
        )
        # print(retrieval_qa_chat_prompt, type(retrieval_qa_chat_prompt))
        combine_docs_chain = create_stuff_documents_chain(self.model, retrieval_qa_chat_prompt)
        rag_chain = create_retrieval_chain(retrieval, combine_docs_chain)

        result = rag_chain.invoke({"input": user_question})

        answer, context, = result["answer"], result["context"]

        metadata = context[0].metadata
        retrieved_document_id = metadata["id"]
        retrieved_document = context[0].page_content

        return QueryResponse(
            question=user_question,
            answer=answer,
            retrieved_document_id=retrieved_document_id,
            retrieved_document=retrieved_document
        )
