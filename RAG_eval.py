# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 openai

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import yaml
import mlflow

rag_chain_config = {
      "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
      "source_table_name": f"{catalog}.{dbName}.{embed_table_name}",
      "vector_search_index_name": f"{catalog}.{dbName}.{embed_table_name}_vs_index",
      "llm_endpoint_name": instruct_endpoint_name,
}

config_file_name = 'rag_chain_config.yaml'
try:
    with open(config_file_name, 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')

# COMMAND ----------

import os

# Specify the full path to the chain notebook
chain_notebook_path = os.path.join(os.getcwd(), "chain_langchain")

# Specify the full path to the config file (.yaml)
config_file_path = os.path.join(os.getcwd(), "rag_chain_config.yaml")

print(f"Chain notebook path: {chain_notebook_path}")
print(f"Chain notebook path: {config_file_path}")

# COMMAND ----------

user_account_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

# Set the experiment name
mlflow.set_experiment(f"/Users/{user_account_name}/rach_rag_experiment")

model_name = f"{catalog}.{dbName}.{registered_model_name}"

# Log the model to MLflow
# TODO: remove example_no_conversion once this papercut is fixed
with mlflow.start_run(run_name="rach_rag_chatbot"):
    # Tag to differentiate from the data pipeline runs
    mlflow.set_tag("type", "chain")

    input_example = {
        "messages": [{"role": "user", "content": "授業時間は一コマどのくらいですか？"}]
    }

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=chain_notebook_path,  # Chain code file e.g., /path/to/the/chain.py
        model_config=config_file_path,  # Chain configuration set in 00_config
        artifact_path="chain",  # Required by MLflow
        input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        registered_model_name=model_name,
    )

# COMMAND ----------

chain = mlflow.langchain.load_model(logged_chain_info.model_uri)

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

model_name = f"{catalog}.{dbName}.{registered_model_name}"
uc_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=model_name)

# COMMAND ----------

### Test the registered model
registered_agent = mlflow.langchain.load_model(f"models:/{model_name}/{uc_model_info.version}")

registered_agent.invoke(input_example)

# COMMAND ----------

# Deploy

import os
import mlflow
from databricks import agents

# modelをdeployする
deployment_info = agents.deploy(
    model_name, 
    uc_model_info.version 
)

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")

review_instructions = """### Rach FAQチャットボットのテスト手順

チャットボットの品質向上のためにぜひフィードバックを提供ください。

1. **多様な質問をお試しください**：
   - 実際のお客様が尋ねると予想される多様な質問を入力ください。これは、予想される質問を効果的に処理できるか否かを確認するのに役立ちます。

2. **回答に対するフィードバック**：
   - 質問の後、フィードバックウィジェットを使って、チャットボットの回答を評価してください。
   - 回答が間違っていたり、改善すべき点がある場合は、「回答の編集（Edit Response）」で修正してください。皆様の修正により、アプリケーションの精度を向上できます。

3. **回答に付随している参考文献の確認**：
   - 質問に対してシステムから回答される各参考文献をご確認ください。
   - Good👍／Bad👎機能を使って、その文書が質問内容に関連しているかどうかを評価ください。

チャットボットの評価にお時間を割いていただき、ありがとうございます。エンドユーザーに高品質の製品をお届けするためには、皆様のご協力が不可欠です。"""

agents.set_review_instructions(model_name, review_instructions)

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
w = WorkspaceClient()
now = time.time()
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

    if time.time() - now > 3600:
        raise Exception("Endpoint did not deploy in 1 hour")

print(f"\n\nReview App: {deployment_info.review_app_url}")

# COMMAND ----------

from databricks import agents

user_list = ["ttc2350sa0009@edu.tech.ac.jp"]
agents.set_permissions(model_name=model_name, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

print(f"Share this URL with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------


