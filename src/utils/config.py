from datetime import datetime
import environ

from src.utils.constants import TOP_K_DEFAULT, RELEVANT_SCORE_DEFAULT



@environ.config(prefix="APP")
class AppConfig:

    @environ.config(prefix="ARXIV")
    class Arxiv:
        max_results = environ.var(help="The number of results to return", default=TOP_K_DEFAULT)
        relevance_score_threshold = environ.var(help="The relevance score threshold to use", default=RELEVANT_SCORE_DEFAULT, converter=int)
        publish_year_threshold = environ.var(help="The publish date's year threshold to use", default=0, converter=int)

    @environ.config(prefix="OPENAI_AZURE")
    class AzureOpenAI:
        url = environ.var(help="Azure OpenAI URL")
        api_key = environ.var(help="Azure OpenAI API key")

    @environ.config(prefix="RESEARCHER")
    class ResearchAgent:
        temperature = environ.var(help="The temperature of the LLM", default=0, converter=float)
        num_of_hypothesis = environ.var(help="The number of hypothesis to return for the query", default=2, converter=int)

    @environ.config
    class Judge:
        name = environ.var(help="The name of the Judge", default="PatronusAI/glider-gguf.glider_Q5_K_M.gguf")
        tokenizer = environ.var(help="The tokenizer of the Judge", default="PatronusAI/glider")
        temperature = environ.var(help="The temperature of the LLM", default=0, converter=float)


    arxiv = environ.group(Arxiv)
    judge = environ.group(Judge)
    openai = environ.group(AzureOpenAI)
    researcher = environ.group(ResearchAgent)

config = AppConfig.from_environ()