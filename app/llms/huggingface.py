from app.vectorstore import PineconeClinet
from app.models import QueryRequest, QueryResponse
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import logger


class HuggingFaceLLM:
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        try:
            chat_model = HuggingFacePipeline.from_model_id(
                model_id='LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct',
                task='text-generation',
                pipeline_kwargs=dict(
                    max_new_tokens=1024,
                    do_sample=False,
                    repetition_penalty=1.03
                ),
                model_kwargs={'quantization_config': quantization_config, "trust_remote_code": True,
                              "device_map": "auto"}
            )
        except Exception as e:
            logger.error("Failed to load LLM:", exc_info=e)
            raise
        self.model = ChatHuggingFace(llm=chat_model, model_id='LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct')

    def generate_answer(self, query: QueryRequest):
        user_question = query.question
        vectorstore = PineconeClinet()
        retrieval = vectorstore.get_retrieval()

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

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
