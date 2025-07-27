# rach

## 動作検証

1. `faq-chatbot.py`を上から実行する
    - 学校HPのスクレイピングと、DB, ベクトルDBのエンドポイント作成プログラム
    - Run allボタンで全実行可能
2. `.env`にCohereのAPI keyを追加する
    - `cp .env.sample .env`をし、COHERE_API_KEYに管理者からもらったAPI keyを配置する
3. `chain_langchain.py`を上から実行する
    - ユーザーからの質問に回答する際のフローをlangchainのLCELを使ってchain化
    - 一番下の `# chain.invoke(input_example)`のコメントを外す
    - Run allボタンで全実行可能

※ `chain_langchain.py`での検証時、エンドポイントがない旨のエラーが出た場合は[./create-vector-db.py](./create-vector-db.py)をすべて実行し、エンドポイントを再度作成してください。

## RAGの定量/定性評価方法

前提: 上記動作検証ができているものとする

### 定性評価方法

- レビューアプリをデプロイし、人間がRAGの回答の評価をすることができます。

#### 実行方法

1. `RAG_eval.py`を実行する
2. `deployment_info.review_app_url`で出力されるurlにアクセスし、レビューする

### 定量評価方法

- LLM-as-a-Judgeを使い、RAGの精度評価をします。
- Mosaic AI Agent Evaluationの機能を使って評価します。

#### 評価基準

以下の4つで評価します。

- Correctness(正確性)：回答は正確か​
- Context sufficiency(文脈充足性)：回答に必要な情報を取得できたか​
- Groundedness(根拠性)：取得した情報に基づいて回答できているか​
- Overall(総合性)：上記三つが全て合格か​

#### 実行方法

1. `RAG_eval-LLM-As-A-Judge.py`を実行する
2. `mlflow.evaluate()`で出力された`View run new_eval_run at:`のurlの`Tracing`タブで評価結果を確認できます。
