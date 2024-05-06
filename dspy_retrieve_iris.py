## This is an unofficial retriever for InterSystems IRIS Vectors built to illustrate how a custom retrieval model can be built for DSPy RAG application.
## Author: Elijah Cotterrell (github.com/ecotterr)
## See https://github.com/stanfordnlp/dspy/blob/main/dspy/retrieve for more examples.
import warnings
from typing import Callable, Optional

import dspy

try:
    import iris
except ImportError:
    raise ImportError(
        """
        The InterSystems IRIS DB-API driver is not installed.
        Install it with 'pip install ./wheels/intersystems_irispython-3.2.0-py3-none-any.whl' 
        or download and install the DB-API driver directly from https://intersystems-community.github.io/iris-driver-distribution/
        """
    )

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    warnings.warn("Sentence Transformers is not installed. Install it with 'pip install sentence-transformers' to use Hugging Face embedding models.")
try:
    import openai
except ImportError:
    warnings.warn("OpenAI is not installed. Install it with 'pip install openai' to use OpenAI embedding models.")


class IRISVectorRM(dspy.Retrieve):

    def __init__(
            self,
            db_args: dict,
            iris_embedding_table_name: str,
            iris_embedding_fields: list[str],
            iris_embedding_key: str,
            iris_data_table_name: Optional[str],
            iris_data_fields: list[str],
            iris_data_key: Optional[str],
            embedding_model: str = "avsolatorio/GIST-Embedding-v0",
            openai_client: Optional[openai.OpenAI] = None,
            embedding_func: Optional[Callable] = None,
            k: int = 10,
            include_similarity: bool = False,
    ):
        assert openai_client or embedding_model or embedding_func, "Either openai_client, Hugging Face embedding_model or embedding_func must be provided."

        self.openai_client = openai_client
        self.embedding_func = embedding_func

        self.conn = iris.connect(**db_args)

        self.iris_embedding_table_name = iris_embedding_table_name
        self.iris_embedding_fields = iris_embedding_fields
        self.iris_embedding_key = iris_embedding_key
        self.iris_data_table_name = iris_data_table_name
        self.iris_data_fields = iris_data_fields
        self.iris_data_key = iris_data_key

        self.embedding_model = embedding_model
        self.include_similarity = include_similarity

        if openai_client is None:
            self.hf_model = SentenceTransformer(embedding_model, cache_folder='.\\huggingface_cache')

        super().__init__(k=k) 

    def forward(self, query: str, k: Optional[int] = None, **kwargs):
        query_embedding = self._get_embeddings(query)

        if not self.k:
            self.k = k

        retrieved_docs = []

        data_fields = "data." + ", data.".join(self.iris_data_fields)
        embedding_field = "vector." + self.iris_embedding_fields[0] # Will need to add support for multiple embedding / vector fields to search on
       
        query = f"""
            SELECT TOP {self.k} {data_fields}
            FROM {self.iris_data_table_name} data
            JOIN {self.iris_embedding_table_name} vector
            ON vector.{self.iris_data_key} = data.{self.iris_embedding_key}
            ORDER BY VECTOR_DOT_PRODUCT(TO_VECTOR({query_embedding}), TO_VECTOR(?)) DESC
        """ 

        cursor = self.conn.cursor()
        cursor.execute(query, [str(query_embedding.tolist())])
        result_set = list(cursor.fetchall())
        for row in result_set:
            data = dict(zip(self.iris_data_fields, row))
            retrieved_docs.append(dspy.Example(**data))

        return retrieved_docs

    def _get_embeddings(self, query: str) -> list[float]:
        if self.openai_client is not None:
            return self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query,
                encoding_format="float"
            ).data[0].embedding
        else:
            if self.hf_model is not None:
                return self.hf_model.encode(query)[0]
            else:
                return self.embedding_func(query)