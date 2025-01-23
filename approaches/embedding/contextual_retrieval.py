# Databricks notebook source
# %pip install -qU  langchain==0.2.11 langchain_core==0.2.23 langchain-community==0.2.9 mlflow databricks-agents databricks-langchain=0.1.1

# %restart_python

# COMMAND ----------

import mlflow

# from langchain_community.chat_models import ChatDatabricks
from databricks_langchain import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)

# COMMAND ----------

def get_model(model_config_path: str) -> ChatDatabricks:
    model_config = mlflow.models.ModelConfig(development_config=model_config_path)
    return ChatDatabricks(
        endpoint=model_config.get("llm_endpoint_name"),
        extra_params={"temperature": 0},
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

def process_and_annotate_document(chunk: str, page_contents: str, model_config_path: str = 'rag_chain_config.yaml') -> list[dict[str, str]]:
    chain = (
        get_prompt_template()
        | get_model(model_config_path)
        | StrOutputParser()
    )

    input_message = {
        'page_contents': page_contents,
        'chunk_content': chunk,
    }
    res = chain.invoke(input_message)
    return res + "\n\n" + chunk

# COMMAND ----------

eg = {'content': 'FUTURE’S VIEW  \nAI技術とロボットを総合的に学ぶ\nAI技術とロボットを総合的に学ぶ\nAIとものづくりのスキルを兼ね備えた総合技術者を目指し、AIスピーカーや自動運転車のようなAI搭載製品に関する実践教育を受けます。IoT技術を用いて社会課題を解決するスペシャリストを育成するプログラムに焦点を当て、プロフェッショナルの指導のもと、ソリューションの設計、実装、制御などを学びます。',
  'page_contents':"FUTURE’S VIEW  \nAI技術とロボットを総合的に学ぶAIとものづくりのスキルを兼ね備えた総合技術者を目指し、AIスピーカーや自動運転車のようなAI搭載製品に関する実践教育を受けます。IoT技術を用いて社会課題を解決するスペシャリストを育成するプログラムに焦点を当て、プロフェッショナルの指導のもと、ソリューションの設計、実装、制御などを学びます。目指せる職業  \nソフトウェア開発  \nプログラマー/Javaプログラマー/ソフトウェアプログラマー  \nシステム設計と開発  \nシステムエンジニア/システムインテグレーター  \nプロジェクト管理  \nプロジェクトマネージャー  \n専門技術分野  \nAIエンジニア/IoTエンジニア/セキュリティエンジニア/ロボットエンジニア  \n就職実績 在学中から卒業後まで、万全のサポートで「夢の実現」へと導きます卒業生の就職状況（2011〜2023年）  \n13年連続全体就職率  \n100  \n%  \n希望業界への就職率  \n91  \n.7  \n%学びのポイントモノづくりからAIプログラミングまで幅広く学べる  \npoint 01  \nモノづくりからAIプログラミングまで幅広く学べる  \npoint 01  \nTECH.C. は世界的ITベンダー企業からの認定されたカリキュラムを導入。 授業では、ロボット技術だけではなく、近年注目を集める機械学習の基本的な操作や機械学習に関する一般的なライブラリ・ツールの使い方を学びます。  \nピックアップ授業  \n組込み/IoT実習  \nIoTでは、物にセンサーを取り付けてデータを収集し、このデータは組み込みシステムで処理されます。カム機構とリンク機構の基本を学び、カム・リンク機構モジュールを実際に組み立てて、その構造や特性を理解することで、効率的な機構設計を目指します。  \n機械学習  \nこの授業では、拡大するビジネス分野のAI利用に対応し、機械学習の基本操作と一般的なツールの使い方、特に需要が高まっているディープラーニングの理論と実践を学びます。  \n機械工作基礎  \n機械工作の基本として、安全管理、加工法、設計、作品製作、寸法測定法を学びます。ボール盤加工、旋盤加工、平面研削加工、フライス盤加工を基本操作で習得し、プラスチックとアルミニウム板材で小箱を設計・製作します。  \nC言語  \nC言語は汎用性が高く、WEBアプリ、スマホアプリ、組み込みアプリの制作が可能です。ビッグデータ解析、AI、VRにおいて大量データの高速処理にも必須です。  \nTECH.LABで工作機械の使い方やものづくり全般を学ぶことが出来る  \npoint 02  \nTECH.LABで工作機械の使い方やものづくり全般を学ぶことが出来る  point 02  \n2019年4月に完成したTECH.C. LAB。工作機械だけではなく、部品や材料加工のための工具も設置。組み立てたロボットのモーション制御やセンサーコントロールなども行えるため、ロボット製作には最高の環境で学ぶことができます。  \n制作事例  \nロボット  \nスマートフォンで操作できるロボットを制作  \n詳しく見る  \n顔認識で追尾するロボット  \n一定距離を保ち顔認証で追尾するロボットを制作しました。  \n詳しく見る  \n自動アルコール  \n自分好みのアルコールを量を出す自動噴射機を作成しました。  \n詳しく見る  \n動作リンクロボットアーム  \n自分脳でとロボットのアームの動きをリンクさせます。  \n詳しく見る  \n閉じる  \n閉じる  \n閉じる  \n閉じる  \n高度専門士が取得してIT大手企業まで目指せる  \npoint 03  \n高度専門士が取得してIT大手企業まで目指せる  \npoint 03  \n4年制の専門学校修了者に付与される4年制大学卒業の学士と同等の称号である「高度専門士」が取得できます。3,400時間以上の授業で技術を確実に身につけることができるため、就職時に有利になる可能性があります。  \n万全のサポート体制  \nインターンシップ  \n就職前の疑問や不安を解消するため、2週間から1ヶ月のプロの現場研修を行い、実際の仕事体験を通して、現場感覚と即戦力となるスキルを身に付けます。  \n詳しく見る  \n合同企業説明会  \n年間100社以上のトップ企業が来校し、会社説明やポートフォリオ審査、就職アドバイスを提供。本校独自の業界との強いつながりにより、1日で複数企業と出会うチャンスがあります。  \n詳しく見る  \nキャリアセンター  \n『夢』を『仕事』として実現する。そのお手伝い、私たちにおまかせください。一人ひとりの希望や能力を把握し、プロのスタッフが的確にアドバイスいたします。いつでもご相談ください。  \n詳しく見るカリキュラム主な授業・カリキュラム  \n基礎カリキュラム  \n資格を学ぶ CompTIA IT Fundamentals / Microsoft Office Specialist / Azure Al Fundamentalsロボット基礎を学ぶ IT基礎 / コンピュータ基礎 / C言語 / ITリテラシー / 物理数学社会人基礎を学ぶ 英会話 / コミュニケーションスキルスペシャルカリキュラム 海外実学研修 / 特別講義 / ロボットゼミ / 企業プロジェクト / ITプロジェクト  \n専門カリキュラム※自分にあった授業を選択できます  \nロボット製作技術を学ぶ 機械設計 / SolidWorks / 機械工作 / 電気電子実習 / Fusion 360機械加工を学ぶ 機械工作基礎 / 機械工作応用 / ロボット製作基礎 / 機械工作実習 / 電子工作実習 / loTプロジェクトで実践する メカトロニクス実践 / ロボット製作応用AI(人工知能を学ぶ) Python / GPU / ディープラーニング / 統計分析 / Raspberry Pi / Jetson Nano / ROS / クラウド  \n就職対策カリキュラム  \n資格取得サポート 就職にも有利になる資格を取得する業界特別ゼミ・特別講義 トッププロの直接指導で技術と知識を学ぶ海外実学研修 世界のトップ企業から業界の最先端を学ぶロボット大会参加 企業見学 企業に直接訪問し業界のイマを学ぶ履歴書作成指導 履歴書・エントリーシートの添削指導模擬面接 実際役に立つ面接指導機械工場見学・実習 ポートフォリオ指導 就職活動で必要となるポートフォリオ制作を実施合同企業説明会 年間120社以上の企業が直接来校し、会社説明を受ける  \n時間割  \n月 火 水 木 金 土 1時間目 - 福祉工学ものづくり - Python応用 - ロボットゼミ 2時間目 英語 福祉工学ものづくり - Python応用 ロボット製作応用（隔週） ロボットゼミ 3時間目 機械学習 ITプロジェクト - Unity開発演習 ロボット製作応用（隔週） ロボット創造実習 4時間目 機械学習 ITプロジェクト - Unity開発演習 ロボット製作応用（隔週） ロボット創造実習 5時間目 Office / MOS - - - ロボット製作応用（隔週） - 6時間目 - - - - - -Wメジャーカリキュラム夢や目標に合わせて、専攻以外の授業も履修可能！  \n入学時に選択した専攻以外に、他の専攻の科目を履修できるのが「Wメジャーカリキュラム」。将来のやりたいことや好奇心に合わせて、興味のある授業を受けることができます。  \n目指せる資格  \nMicrosoft Certified ITサービスマネージャー試験 CompTIA A+ ネットワークスペシャリスト試験 ITパスポート試験 シスコ認定技術者試験（CCNA、CCNP） 情報セキュリティマネジメント試験 CSWA 基本情報技術者試験 CSWP 応用情報技術者試験 Python3エンジニア認定基礎試験 システムアーキテクト試験",
  'url': 'https://www.tech.ac.jp/course/robot/robot-ai/'}

# COMMAND ----------

input_data = {
    "chunk": eg['content'],
    "page_contents": eg['page_contents'],
    "model_config_path": "../../rag_chain_config.yaml"
}
# process_and_annotate_document(**input_data)

# COMMAND ----------


