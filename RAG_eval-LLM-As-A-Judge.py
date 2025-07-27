# Databricks notebook source
# MAGIC %pip install -U -qqqq mlflow==2.19.0 mlflow-skinny==2.19.0 \
# MAGIC     databricks-vectorsearch==0.40 databricks-sdk==0.34.0 databricks-agents==0.13.0 databricks-langchain==0.1.1 \
# MAGIC     langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 \
# MAGIC     cohere==5.13.0 python-dotenv==0.21.1
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os

# Specify the full path to the chain notebook
chain_notebook_path = os.path.join(os.getcwd(), "chain_langchain.py")

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

model_name = f"{catalog}.{dbName}.{registered_model_name}"
uc_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=model_name)

# COMMAND ----------

import mlflow
import pandas as pd
exsample_eval_set  = [
    {
      "request_id": "1",
      "request": "AO入学はありますか？",  # question
      "expected_response": "はい、あります！AO入学制度は、学力試験だけでは評価できない個性や意欲を重視した入学選考方法です。"  # 模範解答
    },
]

#### Convert dictionary to a pandas DataFrame
csv_path = "eval-dataset.csv"
eval_set_df = pd.read_csv(csv_path)

eval_set_df["request_id"] = eval_set_df["request_id"].astype(str)  # 文字列じゃないとエラー出る

model_name = f"{catalog}.{dbName}.{registered_model_name}"

###
# mlflow.evaluate() call
###
# with mlflow.start_run(run_id=logged_chain_info.run_id):
with mlflow.start_run(run_name="new_eval_run"):
  evaluation_results = mlflow.evaluate(
      data=eval_set_df,
      # data=exsample_eval_set,
      model=f"models:/{model_name}/{uc_model_info.version}",
      model_type="databricks-agent",
  )

# COMMAND ----------

evaluation_results.tables["eval_results"]

# COMMAND ----------

# import requests
# import json

# data = {
#   "messages": [{"role": "user", "content": "エアコンの買い換えを決める際の判断基準はありますか？"}]
# }

# databricks_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
# databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# headers = {"Context-Type": "text/json", "Authorization": f"Bearer {databricks_token}"}


# url = "https://adb-241438725865650.10.azuredatabricks.net/ml/review/dev.rach_db.rach_chatbot_model/8/instructions"
# response = requests.post(
#     url=url, json=data, headers=headers
# )

# print(response.json()["choices"][0]["message"]["content"])

# COMMAND ----------


