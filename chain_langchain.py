# Databricks notebook source
# MAGIC %pip install databricks-langchain=0.1.1
# MAGIC %pip install cohere
# MAGIC %pip install mlflow lxml==4.9.3 transformers==4.30.2 databricks-vectorsearch==0.38 databricks-sdk==0.28.0 databricks-feature-store==0.17.0 langchain==0.2.11 langchain_core==0.2.23 langchain-community==0.2.9 databricks-agents
# MAGIC %pip install python-dotenv
# MAGIC # %pip install databricks-langchain langchain==0.2.11 langchain-core==0.2.23 langchain-community==0.2.9

# COMMAND ----------

# MAGIC %pip install databricks-agents mlflow mlflow-skinny databricks-vectorsearch

# COMMAND ----------

# databricksのpythonを再起動させる
dbutils.library.restartPython()

# COMMAND ----------

from operator import itemgetter
import mlflow
import os

from databricks.vector_search.client import VectorSearchClient

from langchain_community.chat_models import ChatDatabricks
# from databricks_langchain import ChatDatabricks
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

from langchain.schema import Document
from typing import Optional, Dict, Any, List

# scoreを返したいので、独自に実装する
# .as_retrieverでやると、similarity_search が内部で呼ばれるため、scoreが返ってくる similarity_search_with_scoreを呼ぶようにしている
# https://qiita.com/Oxyride/items/ac7e32714f5fa673d9e4
class CustomDatabricksVectorSearch(DatabricksVectorSearch):
    # search_type: simirarity
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        *,
        query_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_with_score = self.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            query_type=query_type,
            **kwargs,
        )
        for doc, score in docs_with_score:
            # 類似度スコアを保存する
            doc.metadata['score'] = score
        return [doc for doc, _ in docs_with_score]

    # search_type: similarity_score_threshold
    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """
        Default similarity search with relevance scores. Modify if necessary
        in subclass.
        Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        docs_and_similarity_scores = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]
        for doc, score in docs_and_similarity_scores:
            # 類似度スコアを保存する
            doc.metadata['score'] = score
        return docs_and_similarity_scores


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

vs_index = vs_client.get_index(
    endpoint_name=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("vector_search_index_name")
)

############
# Turn the Vector Search index into a LangChain retriever
############
vector_search_as_retriever = CustomDatabricksVectorSearch(
    vs_index,
    text_column="content",
    columns=[
        "id",
        "content",
        "url",
    ],
).as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        'score_threshold': 0.7,
        'query_type': 'hybrid',
        'k': 20,
    }
)

############
# Required to:
# 1. Enable the RAG Studio Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
############

mlflow.models.set_retriever_schema(
    primary_key="id",
    text_column="content",
    doc_uri="url",  # Review App uses `doc_uri` to display chunks from the same document in a single view
    other_columns=["score"],
)


############
# Method to format the docs returned by the retriever into the prompt
############
def format_context(docs: list[Document]) -> str:
    chunk_template = "\nPassage: {chunk_text}\n"
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            document_uri=d.metadata["url"],
            score=d.metadata["score"],
        )
        for d in docs
    ]

    return "".join(chunk_contents)

# COMMAND ----------

# vector_search_as_retriever.invoke("授業時間は一コマどのくらいですか？")

# COMMAND ----------

############
# Prompt Template for generation
############
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            """あなたは東京デザインテクノロジーセンター専門学校（通称TECH.C.）の公式チャットbotです。以下の【参考情報】を基に、ユーザーからの【質問】に対して正確で簡潔な回答を行ってください。

- 【参考情報】以外の情報には基づかずに回答してください。
- 【参考情報】に該当がない場合や不明確な場合は、「申し訳ありませんが、その質問にお答えできる情報がありません。」と答えてください。
- 必要に応じて、ユーザーが質問を明確化できるように助言を行ってください。

回答の語調はフレンドリーかつ丁寧に保ち、ユーザーが気軽に質問できる雰囲気を大切にしてください。"""
        ),
        # User's question
        ("user", """【参考情報】\n{context}\n\n【質問】\n{question}"""),
    ]
)

# 一般質問用の簡易プロンプト
no_content_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーからの質問に答えてください。"),
        ("user", """【質問】\n{question}"""),
    ]
)


# COMMAND ----------

# classification_prompt = ChatPromptTemplate.from_messages(
#     [
#         (  # System prompt contains the instructions
#             "system",
#             """You are an AI assistant tasked with classifying questions into two categories: 'general' or 'specific'.

# General: The question asks for common knowledge, general definitions, or broad explanations.

# Your task is to classify the following question strictly as either 'general' or 'specific'.
# The answer must be exactly one of these two words: 'general' or 'specific'.
# Do not provide any explanations, additional text, or alternative formats.

# Classify the following question:
# **Answer only with 'general' or 'specific'.** e.g. 'general'"""
#         ),
#         (
#             'user', "Question: {question}"
#         )
#     ]
# )


# COMMAND ----------

classification_prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            """You are an AI assistant tasked with classifying questions into two categories: 'general' or 'specific'.

1. General: The question asks for common knowledge, general definitions, or broad explanations.
Examples
    - What is artificial intelligence?
    - How does a neural network work?

2. Specific: The question requires information from a specific domain, dataset, or context.
    - Questions that require specific data or examples.
    - Questions related to schools, education, or academic topics.
Examples
    - How many students are enrolled?
    - How many years does the school have?

Classify the following question:
**Answer only with 'general' or 'specific'.**"""
        ),
        (
            'user', "Question: {question}"
        )
    ]
)


# COMMAND ----------

############
# FM for generation
############
model = ChatDatabricks(
    endpoint=model_config.get("llm_endpoint_name"),
    extra_params={"temperature": 0.7, "max_tokens": 1500},
)

# COMMAND ----------

# 一般質問かどうかを判定するchain
classification_chain = (
    classification_prompt
    | ChatDatabricks(endpoint=model_config.get("llm_mini_endpoint_name"), extra_params={"temperature": 0, "max_tokens": 5})
    | StrOutputParser()
)

# COMMAND ----------

from typing import Optional
def select_prompt(context: Optional[str]) -> ChatPromptTemplate:
    """
    LLMを使って質問を分類し、それに応じてプロンプトを選択。
    """
    if context is None:
        # Retrieverがスキップされた場合、一般質問用のプロンプトを使用
        return no_content_prompt
    return rag_prompt


# COMMAND ----------

def is_general_question(question: str) -> bool:
    # LLMを使って質問を分類
    classification_result = classification_chain.invoke({"question": question}).strip().lower()
    return classification_result == "general"


# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.vectorstores.base import VectorStoreRetriever
from mlflow.tracing.constant import SpanAttributeKey
import json
from mlflow.entities import SpanType

# mlflowとMosaic AI Agent Evaluation (DatabricksのLLM-as-a-Judge) の仕様上、
# Context Sufficient と Groundedness の評価に使われるdocsは、
# mlflowでRETRIEVER属性が付いたspanの中でも最後にトレースされたものが対象となる。
#
# この関数を使わない場合、HyDEで取得したdocsのみが「回答に使用するために取得したdocs」と判定されてしまう。
# そこで、新たにRETRIEVER属性を持つspanを作成し、すべてのdocsを含めることで、
# 正しく評価対象として認識されるようにする。
def set_retrieved_documents_for_mlflow(docs: list[Document]) -> None:
    with mlflow.tracing.fluent.start_span(
        name="final_retrieved_docs",
        span_type=SpanType.RETRIEVER
    ) as retrieval_span:
        retrieval_span.set_attribute(SpanAttributeKey.OUTPUTS, docs)

def parallel_retrieval_grouped(queries: list[str], retriever) -> list[Document]:
    """各クエリに対して retriever.invoke を並列実行して結果を集約する"""
    all_docs = []
    with ThreadPoolExecutor() as executor:
        # クエリの並列処理は、mlflowのtraceが複数にまたがってしまうので非常によろしくないが、並列だと3s短縮されるのでこちらのメリットの方が大きいと判断
        futures = {executor.submit(retriever.invoke, q): q for q in queries if q != ''}
        for future in as_completed(futures):
            docs = future.result()
            all_docs.extend(docs)
        
    return all_docs


# COMMAND ----------

def merge_and_sort_docs(docs_dict: dict) -> dict:
    """
    docs_dict は
      {
          "retriever_docs": [...],
          "hyde_docs": [...]
      }
    の形式で、両方の結果をマージして重複除去、スコアで降順ソートした docs を
    辞書として { "docs": unique_docs } で返す
    """
    retriever_docs = docs_dict.get("retriever_docs", [])
    hyde_docs = docs_dict.get("hyde_docs", [])
    all_docs = retriever_docs + hyde_docs
    # 重複除去（page_content をキーに）
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    return {"docs": unique_docs}

# COMMAND ----------

import cohere
import os
from dotenv import load_dotenv

if "COHERE_API_KEY" not in os.environ:
    load_dotenv()

rerank_model = cohere.ClientV2(os.environ["COHERE_API_KEY"],)

def rerank_docs(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:

    docs_content = [d.page_content for d in docs]

    results = rerank_model.rerank(
        query=query,
        documents=docs_content,
        top_n=top_n,
        model="rerank-v3.5",
    ).results

    reranked_docs = []
    for reranked_doc_idx_and_score in results:
        reranked_doc_idx = reranked_doc_idx_and_score.index
        docs[reranked_doc_idx].metadata["relevance_score"] = reranked_doc_idx_and_score.relevance_score
        reranked_docs.append(docs[reranked_doc_idx])

    set_retrieved_documents_for_mlflow(reranked_docs)

    return reranked_docs

# COMMAND ----------

from langchain.retrievers import RePhraseQueryRetriever

# HyDEプロンプトテンプレート
hyde_prompt_template = """ \
以下の質問の回答を書いてください。
質問: {question}
回答: """

mini_model = ChatDatabricks(
    endpoint=model_config.get("llm_mini_endpoint_name"),
    extra_params={"temperature": 0.7, "max_tokens": 1500},
)

# HyDE Prompt
hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template)

# HyDE retriever
rephrase_retriever = RePhraseQueryRetriever.from_llm(
    retriever = vector_search_as_retriever,
    llm = mini_model,
    prompt = hyde_prompt,
)

# COMMAND ----------

rewrite_prompt_template = """

あなたは、検索エンジンの精度を向上させるAIアシスタントです。
ユーザーが入力したクエリをもとに、より効果的な検索を行うためのバリエーションを作成してください。
質問には答えず、バリエーションを作ることに専念してください。

質問: {original_query}

- 言い換え（3つ）
- シンプルな要約表現
- より一般的な表現（1つ）
- より専門的な表現（1つ）
- 詳細化したバージョン（1つ）

出力は必ず**カンマ(',')区切り**で記述してください。
例: 要約, 言い換え1, 言い換え2, 言い換え3, 一般向け, 専門的, 詳細版
"""
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt_template)

rewrite_chain = (
    rewrite_prompt
    | mini_model
    | StrOutputParser()
)


# 質問のre-write
def rewrite_question(question: str) -> list[str]:
    response = rewrite_chain.invoke({"original_query": question})
    try:
        query = response.split(",")
        return [question] + query
    except:
        return [question]

# COMMAND ----------

from langchain_core.runnables import RunnableParallel

parallel_docs_chain = (
    RunnableParallel({
        "retriever_docs": RunnableLambda(
            lambda inputs: parallel_retrieval_grouped(inputs["queries"], vector_search_as_retriever)
        ),
        "hyde_docs": RunnableLambda(
            lambda inputs: rephrase_retriever.invoke({"question": inputs["question"]})
        )
    })
    | RunnableLambda(merge_and_sort_docs)
)

chain = (
    {
        # ユーザーの質問抽出
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        # HyDE用などの複数クエリ生成（例：リライト処理）
        "queries": itemgetter("messages")
                   | RunnableLambda(extract_user_query_string)
                   | RunnableLambda(lambda question: rewrite_question(question))
    }
    | RunnableLambda(
         # 一般的な質問の場合は検索をスキップ（docs を None に設定）
         # そうでなければ、以降の並列検索チェーンを実行して取得した docs を後でマージする
         lambda inputs: (
             {**inputs, "docs": None} if is_general_question(inputs["question"])
             else {**inputs, "docs": parallel_docs_chain.invoke(inputs)["docs"]}
         )
       )
    | RunnableLambda(
         # 取得した docs が存在する場合、再ランキングを実施
         lambda inputs: {**inputs, "docs": rerank_docs(inputs["question"], inputs["docs"])}
         if inputs.get("docs") is not None else inputs
       )
    | RunnableLambda(
         # 再ランキング後の docs を用いて、最終的な文脈 (context) を生成
         lambda inputs: {**inputs, "context": format_context(inputs["docs"])}
         if inputs.get("docs") is not None else {**inputs, "context": None}
       )
    | RunnableLambda(
         # プロンプト選択。ここでは inputs に "question" と "context" があることを前提とする
         lambda inputs: select_prompt(context=inputs["context"]).format(
             question=inputs["question"], context=inputs["context"]
         )
       )
    | model
    | StrOutputParser()
)


mlflow.models.set_model(model=chain)

# COMMAND ----------

input_example = {
  "messages": [{"role": "user", "content": "授業時間は一コマどのくらいですか"}]
#   "messages": [{"role": "user", "content": "プログラマとは？"}]
}

chain.invoke(input_example)

# COMMAND ----------


