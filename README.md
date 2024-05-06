# (Unofficial) DSPy retriever for InterSystems IRIS

```
from dspy_retrieve_iris import IRISVectorRM

iris_retriever = IRISVectorRM(
    db_args = {'hostname':'localhost', 'port':51776, 'namespace':'VECTOR', 'username':keys.db_user, 'password':keys.db_password},
    iris_embedding_table_name = 'RAG_COQA.QandA',
    iris_embedding_fields = ['QuestionEmbedding'],
    iris_embedding_key = 'StoryId',
    iris_data_table_name = 'RAG_COQA.Story',
    iris_data_fields = ['Story', 'Source'],
    iris_data_key = 'ID',
    embedding_model = 'avsolatorio/GIST-Embedding-v0',
    k = 10,
    include_similarity = False
)

import dspy

gpt_turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(
    lm=gpt_turbo, 
    rm=iris_retriever
)

```