import asyncio

from src.application.papers_rag import get_interesting_papers
from src.domain.queries import Query
from src.monitoring.opik import configure_opik
from src.utils.constants import REPORT_FILE_PATH

async def main():
    configure_opik()
    query = Query.from_str("I'm looking for papers on the cybersecurity attacks on agents or agentic systems")

    report = await get_interesting_papers(query)

    with open(REPORT_FILE_PATH, 'w') as f:
        f.write(str(report))

    print(report)
    return report



if __name__ == '__main__':
    asyncio.run(main())
