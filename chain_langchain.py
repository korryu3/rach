# Databricks notebook source
# MAGIC %pip install databricks-agents mlflow mlflow-skinny databricks-vectorsearch

# COMMAND ----------

# MAGIC %pip install langchain==0.2.11 langchain-core==0.2.23 langchain-community==0.2.9

# COMMAND ----------

# databricksのpythonを再起動させる
dbutils.library.restartPython()

# COMMAND ----------

import time
def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
  try:
    dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
  except Exception as e:
    if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
      print(f'Unexpected error describing the index. This could be a permission issue.')
      raise e
  return dict_vsindex.get('status').get('ready', False)

def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

def create_index(vsc, vector_search_index_name, source_table_name):
  # embeddingモデル名
  embedding_endpoint_name = "databricks-gte-large-en"

  #インデックスを新規作成
  print(f"Creating index {vector_search_index_name} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
  endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
  index_name=vector_search_index_name,
  pipeline_type="TRIGGERED",
  source_table_name=source_table_name,
  primary_key="id",
  embedding_source_column="response",
  embedding_model_endpoint_name=embedding_endpoint_name
  )

  #インデックスの準備ができ、すべてエンベッディングが作成され、インデックスが作成されるのを待ちましょう。
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vector_search_index_name)
  print(f"index {vector_search_index_name} on table {source_table_name} is ready")



# COMMAND ----------

from operator import itemgetter
import mlflow
import os

from databricks.vector_search.client import VectorSearchClient

from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)

## Enable MLflow Tracing
mlflow.langchain.autolog()


############
# Helper functions
############
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]


# FIT AND FINISH: We should not require a value here.
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

############
# Connect to the Vector Search Index
############
vs_client = VectorSearchClient(disable_notice=True)

VECTOR_SEARCH_ENDPOINT_NAME = model_config.get("vector_search_endpoint_name")

if VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vs_client.list_endpoints().get('endpoints', [])]:
    vs_client.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vs_client, VECTOR_SEARCH_ENDPOINT_NAME)

if not index_exists(vs_client, VECTOR_SEARCH_ENDPOINT_NAME, model_config.get("vector_search_index_name")):
    create_index(vs_client, model_config.get("vector_search_index_name"), model_config.get("source_table_name"))

vs_index = vs_client.get_index(
    endpoint_name=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("vector_search_index_name")
)

############
# Turn the Vector Search index into a LangChain retriever
############
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column="response",
    columns=[
        "id",
        "query",
        "response"
        # "url",
    ],
).as_retriever(search_kwargs={"k": 5, "query_type": "ann"})

############
# Required to:
# 1. Enable the RAG Studio Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
############

mlflow.models.set_retriever_schema(
    primary_key="id",
    text_column="response",
    # doc_uri="url",  # Review App uses `doc_uri` to display chunks from the same document in a single view
)


############
# Method to format the docs returned by the retriever into the prompt
############
def format_context(docs):
    chunk_template = "Passage: {chunk_text}\n"
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            # document_uri=d.metadata["url"],
        )
        for d in docs
    ]
    return "".join(chunk_contents)


# COMMAND ----------

############
# Prompt Template for generation
############
prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            "【参考情報】のみを参考にしながら【質問】にできるだけ正確に答えてください。わからない場合や、質問が適切でない場合は、分からない旨を答えてください。【参考情報】に記載されていない事実を答えるのはやめてください。",
        ),
        # User's question
        ("user", """【参考情報】
{context}

【質問】
{question}"""),
    ]
)


# COMMAND ----------

############
# FM for generation
############
model = ChatDatabricks(
    endpoint=model_config.get("llm_endpoint_name"),
    extra_params={"temperature": 0.01, "max_tokens": 1500},
)

############
# RAG Chain
############
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)

## Tell MLflow logging where to find your chain.
# `mlflow.models.set_model(model=...)` function specifies the LangChain chain to use for evaluation and deployment.  This is required to log this chain to MLflow with `mlflow.langchain.log_model(...)`.

mlflow.models.set_model(model=chain)

# COMMAND ----------

input_example = {
  "messages": [{"role": "user", "content": "授業時間は一コマどのくらいですか？"}]
}

chain.invoke(input_example)

# COMMAND ----------


