# Databricks notebook source
catalog = "dev"
dbName = db = "rach_db"
volume = "raw_data"
raw_data_table_name = "raw_query"
embed_table_name = "rach_documentation"
registered_model_name = "rach_chatbot_model"

VECTOR_SEARCH_ENDPOINT_NAME="vs_endpoint"
embedding_endpoint_name = "multilingual-e5-large-embedding"
instruct_endpoint_name = "aoai-gpt-4o"

databricks_token_secrets_scope = "rach"
databricks_token_secrets_key = "databricks_token"
databricks_host_secrets_scope = "rach"
databricks_host_secrets_key = "databricks_host"

print('VECTOR_SEARCH_ENDPOINT_NAME =',VECTOR_SEARCH_ENDPOINT_NAME)
print('catalog =',catalog)
print('dbName =',dbName)
print('volume =',volume)
print('raw_data_table_name =',raw_data_table_name)
print('embed_table_name =',embed_table_name)
print('registered_model_name =',registered_model_name)
print('embed_table_name =',embed_table_name)
print('embedding_endpoint_name =',embedding_endpoint_name)
print('instruct_endpoint_name =',instruct_endpoint_name)
