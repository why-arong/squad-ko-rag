# from app.vectorstore import get_relevant_documents
# from chains import get_query_filter_chain
from langchain_upstage import ChatUpstage


def generate_answer(question: str) -> str:

    # filtered_question = get_query_filter_chain(question)
    llm = ChatUpstage()

    return llm.invoke("hi").content
