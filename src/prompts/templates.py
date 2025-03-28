from .base import PromptTemplateFactory
from ..domain.document import Document


class QueryBuilderPromptTemplate(PromptTemplateFactory):
    prompt: str = """
You are the Query Builder agent. Your task is to generate the most accurate and effective search query for retrieving relevant documents from {platform}.

**Input:**

* **Previous Successful Query:** A query that has previously yielded relevant results on {platform}. Use this as a guide for query structure and syntax.
* **Original Query:** The initial search query that was used to retrieve the provided documents.
* **User's Interest:** The specific field or topic the user is interested in.
* **Positive Examples (Optional):** A list of documents that were retrieved by the original query and deemed relevant by the user. Each entry will include the document's title and a brief summary.
* **Negative Examples (Optional):** A list of documents that were retrieved by the original query and deemed irrelevant by the user. Each entry will include the document's title and a brief summary.

**Instructions:**

1.  **Analyze the Previous Successful Query:** Pay close attention to the structure, keywords, and any specific syntax used in the provided successful query. This will serve as a template for constructing the new query.
2.  **Analyze the Original Query:** Understand the terms and concepts used in the original query.
3.  **Prioritize Document Analysis:** If the provided positive and negative examples reveal a dominant subject or domain that deviates significantly from the user's initial interest or the original query, prioritize the analysis of these documents. The goal is to maximize relevance based on user feedback.
4.  **Contextual Enrichment:** Analyze the positive examples to extract the core domain and subject matter that defines the documents. Identify key concepts, keywords, and related terms. Think about adding them to the query if not present.
5.  **Negative Example Filtering:** Use the negative examples to identify terms or concepts that should be excluded from the query.
6.  **Query Optimization:** Construct a query that:
    * Mirrors the structure and syntax of the previous successful query.
    * Is precise and specific.
    * Uses relevant keywords and phrases.
    * Considers synonyms and related terms.
    * Avoids ambiguous terms.
7.  **Output:** Return only the optimized search query.
        
    Positive Examples: {positives}
    
    Negative Examples: {negatives}
    
    User's Interest:
    {interests}
    """

    def create_template(self, fields: str, platform: str,
                        positives: list[Document] = None,
                        negatives: list[Document] = None) -> str:


        if not positives:
            positives_str = []
        else:
            positives_str = "".join([f"\n\t- {doc.content}" for doc in positives])
        if not negatives:
            negatives_str = []
        else:
            negatives_str = "".join([f"\n\t- {doc.content}" for doc in negatives])

        return self.prompt.format(interests=fields, platform=platform,
                                  positives=positives_str, negatives=negatives_str).strip()