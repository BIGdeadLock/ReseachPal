

from arxiv import Client
import asyncio

from src.tools.papers.scorer import LlmAsJudge
from src.agent import research_agent, ResearchDependencies


async def main():

    deps = ResearchDependencies(
        client=Client(), paper_scorer=LlmAsJudge()
    )
    result = await research_agent.run(
        'Give me research papers that mention the use of LLM in the cybersecurity world.'
        , deps=deps
    )
    print('Response:', result.data)


if __name__ == '__main__':
    asyncio.run(main())
