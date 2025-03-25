from .base import PromptTemplateFactory


class QueryBuilderPromptTemplate(PromptTemplateFactory):
    prompt: str = """
    You are an AI language model assistant. 
    ## Task: Your task is to generate a query based on the user fields of interest to search for related {platform}.
    Fields of Interest: {fields}

    ## Constraints:
    - Make it concise.
    - Use the platform in the query.
    - In case the platform is arxiv, just return a short query from the fields. For example: llm agents cybersecurity
    - Output ONLY the query and nothing more!

    Query:
    """

    def create_template(self, fields: str, platform: str) -> str:
        return self.prompt.format(platform=platform, fields=fields).replace("\n", "")

