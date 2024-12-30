import asyncio

import opik
from loguru import logger
from pydantic import PrivateAttr
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from src.domain.prompt import Prompt
from src.domain.queries import Query
from src.domain.state import Query
from src.utils.constants import OPENAI_GPT4O_DEPLOYMENT_ID as GPT4, OPENAI_API_GPTO_VERSION as API_VER
from src.utils.config import config
from src.monitoring.opik import configure_opik
from langgraph.checkpoint.memory import MemorySaver

ques_prompt = Prompt.from_template("""
Given the current user query: {query}, You can ask him one question in order to better understand what to search for,
for example: specific sub domains in the query domain's, specific name of papers, etc.

Write a question to get the answer you need from the user. 
""")

ans_prompt = Prompt.from_template("""
This is the original user's query: {query}.
This is a question the user was asked on the query: {question}.
This is the user's answer: {answer}.

Please rewrite the query to reflect the new details the user wants from the answer to the question.
""")
class HumanInTheLoop:
    name: str = "ask_human_a_question"
    _llm: PrivateAttr()

    def __init__(self):
        self._llm = AzureChatOpenAI(
            azure_endpoint=config.openai.url,
            openai_api_version=API_VER,
            openai_api_key=config.openai.api_key,
            deployment_name=GPT4,
            temperature=0.5
        )

    @opik.track(name="HumanInTheLoop.ask")
    async def ask(self, state: Query):
        """
        Use to ask the user a question to better understand he's, or she's query. You can use it to ask
        for more specific details like subdomains, interest, etc. After the user will give he's, or she's answer,
        the query will be rewritten to reflect on the new changes.
        """
        chain = ques_prompt.generate_prompt() | self._llm | StrOutputParser()
        question = await chain.ainvoke(dict(query=state.query))
        answer = input(question)

        chain = ans_prompt.generate_prompt() | self._llm | StrOutputParser()
        new_query = await chain.ainvoke(dict(query=state.query, question=question, answer=answer))

        # Update the state with the human's input or route the graph based on the input.
        return {
            "query": new_query
        }


if __name__ == '__main__':
    configure_opik()
    checkpointer = MemorySaver()

    query = Query.from_str("Give me research papers that mention the use of LLM in the cybersecurity world.")
    human = HumanInTheLoop()
    graph = StateGraph(Query)
    graph.add_node(human.name, human.ask)
    graph.add_edge(START, human.name)
    graph.add_edge(human.name, END)
    agent = graph.compile(checkpointer=checkpointer)
    loop = asyncio.get_event_loop()
    thread_config = {"configurable": {"thread_id": "some_id"}}
    res = loop.run_until_complete(agent.ainvoke(Query(query=query), config=thread_config))
    logger.info(res['query'])