from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate


def get_query_filter_chain(query: str):
    # TODO: should add query filter node
    return str


# def get_llm():
#     llm = ChatUpstage()
#     return llm
#
#
# def get_rag_chain():
#     llm = get_llm()
#     # example_prompt = ChatPromptTemplate.from_messages(
#     #     [
#     #         ("human", "{input}"),
#     #         ("ai", "{answer}"),
#     #     ]
#     # )
#     # TODO: add context later
#     return llm.invoke("hi")


