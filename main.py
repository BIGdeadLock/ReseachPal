

import asyncio
from loguru import logger

from src.agent.graph import agent
from src.domain.state import OverallState
from src.domain.queries import Query
from src.utils.opik_utils import configure_opik

async def main():
    configure_opik()
    query = Query.from_kw(["LLM", "Cybersecurity"])
    init_state = OverallState(query=query)

    result = await agent.ainvoke(init_state)
    logger.info(result)
    return result



if __name__ == '__main__':
    asyncio.run(main())
