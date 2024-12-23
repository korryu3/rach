# Databricks notebook source
# %pip install -qU  langchain==0.2.11 langchain_core==0.2.23 langchain-community==0.2.9 mlflow databricks-agents

# %restart_python

# COMMAND ----------

import mlflow

from langchain_community.chat_models import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)

# COMMAND ----------

def get_model(model_config_path: str) -> ChatDatabricks:
    model_config = mlflow.models.ModelConfig(development_config=model_config_path)
    return ChatDatabricks(
        endpoint=model_config.get("llm_endpoint_name"),
        extra_params={"temperature": 0, "max_tokens": 1500},
    )

# COMMAND ----------

# 英語版
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (  # System prompt contains the instructions
#             "system",
#             "{doc_content}",
#         ),
#         # User's question
#         ("user", """Here is the chunk we want to situate within the whole document

# {chunk_content}

# Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
# Answer only with the succinct context and nothing else.
# """),
#     ]
# )


# COMMAND ----------

def get_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
    [
        (  # システムプロンプトに指示内容を含める
            "system",
            "{page_contents}",
        ),
        # ユーザーの質問
        ("user", """以下はドキュメント全体の中で位置付けるべきチャンクです。

{chunk_content}

このチャンクをドキュメント全体の中で位置付けるための短く簡潔な文脈を作成してください。
回答は簡潔な文脈のみで、それ以外は含めないでください。
"""),
    ]
)


# COMMAND ----------

import copy
from tqdm import tqdm

def process_and_annotate_documents(split_documents: list[dict[str, str]], model_config_path: str = 'rag_chain_config.yaml') -> list[dict[str, str]]:
    chain = (
        get_prompt_template()
        | get_model(model_config_path)
        | StrOutputParser()
    )

    def process_document(doc: dict[str, str]) -> dict[str, str]:
        page_contents = doc['page_contents']
        input_message = {
            'page_contents': page_contents,
            'chunk_content': doc['content'],
        }
        res = chain.invoke(input_message)
        tmp_doc = copy.copy(doc)
        tmp_doc['content'] = res + "\n\n" + tmp_doc['content']
        return tmp_doc

    processed_documents = [process_document(doc) for doc in tqdm(split_documents)]

    return processed_documents

# COMMAND ----------

eg = [{'content': 'FUTURE’S VIEW  \nAI技術とロボットを総合的に学ぶ\nAI技術とロボットを総合的に学ぶ\nAIとものづくりのスキルを兼ね備えた総合技術者を目指し、AIスピーカーや自動運転車のようなAI搭載製品に関する実践教育を受けます。IoT技術を用いて社会課題を解決するスペシャリストを育成するプログラムに焦点を当て、プロフェッショナルの指導のもと、ソリューションの設計、実装、制御などを学びます。',
  'page_contents':'FUTURE’S VIEW  \nAI技術とロボットを総合的に学ぶ\nAI技術とロボットを総合的に学ぶ\nAIとものづくりのスキルを兼ね備えた総合技術者を目指し、AIスピーカーや自動運転車のようなAI搭載製品に関する実践教育を受けます。IoT技術を用いて社会課題を解決するスペシャリストを育成するプログラムに焦点を当て、プロフェッショナルの指導のもと、ソリューションの設計、実装、制御などを学びます。目指せる職業  \nソフトウェア開発  \nプログラマー/Javaプログラマー/ソフトウェアプログラマー  \nシステム設計と開発  \nシステムエンジニア/システムインテグレーター  \nプロジェクト管理  \nプロジェクトマネージャー  \n専門技術分野  \nAIエンジニア/IoTエンジニア/セキュリティエンジニア/ロボットエンジニア  \n就職実績 在学中から卒業後まで、万全のサポートで「夢の実現」へと導きます\n就職実績\n卒業生の就職状況（2011〜2023年）  \n13年連続全体就職率  \n100  \n%  \n希望業界への就職率  \n91  \n.7  \n%\n\n学びのポイント',
  'url': 'https://www.tech.ac.jp/course/robot/robot-ai/'},
 {'content': '目指せる職業  \nソフトウェア開発  \nプログラマー/Javaプログラマー/ソフトウェアプログラマー  \nシステム設計と開発  \nシステムエンジニア/システムインテグレーター  \nプロジェクト管理  \nプロジェクトマネージャー  \n専門技術分野  \nAIエンジニア/IoTエンジニア/セキュリティエンジニア/ロボットエンジニア  \n就職実績 在学中から卒業後まで、万全のサポートで「夢の実現」へと導きます\n就職実績\n卒業生の就職状況（2011〜2023年）  \n13年連続全体就職率  \n100  \n%  \n希望業界への就職率  \n91  \n.7  \n%\n\n学びのポイント',
    'page_contents':'FUTURE’S VIEW  \nAI技術とロボットを総合的に学ぶ\nAI技術とロボットを総合的に学ぶ\nAIとものづくりのスキルを兼ね備えた総合技術者を目指し、AIスピーカーや自動運転車のようなAI搭載製品に関する実践教育を受けます。IoT技術を用いて社会課題を解決するスペシャリストを育成するプログラムに焦点を当て、プロフェッショナルの指導のもと、ソリューションの設計、実装、制御などを学びます。目指せる職業  \nソフトウェア開発  \nプログラマー/Javaプログラマー/ソフトウェアプログラマー  \nシステム設計と開発  \nシステムエンジニア/システムインテグレーター  \nプロジェクト管理  \nプロジェクトマネージャー  \n専門技術分野  \nAIエンジニア/IoTエンジニア/セキュリティエンジニア/ロボットエンジニア  \n就職実績 在学中から卒業後まで、万全のサポートで「夢の実現」へと導きます\n就職実績\n卒業生の就職状況（2011〜2023年）  \n13年連続全体就職率  \n100  \n%  \n希望業界への就職率  \n91  \n.7  \n%\n\n学びのポイント',
  'url': 'https://www.tech.ac.jp/course/robot/robot-ai/'},
]

# COMMAND ----------

# process_and_annotate_documents(eg, '../../rag_chain_config.yaml')

# COMMAND ----------


