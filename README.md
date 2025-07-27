# Rach

## 📘 概要

**RAGの実用化課題と品質向上へのアプローチ​**

- 課題：多くの企業がRAGを導入するも、本番環境での運用が困難​
- 目的：実践的なRAGを実現するための品質向上ノウハウを学ぶ​
- 方法： Azure Databricksの機能や最新研究を活用し、品質を向上​
- 目標：HP上で運用可能なレベルのRAGを構築​

## 📊 精度比較まとめ

|手法|簡易説明|精度|上記手法との精度差|
|---|---|---|---|
|ベースライン|学校HPの情報を元にRAGを実行|16%|--|
|+データ追加|HP以外の独自データを追加|51%|+35%|
|+Contextual Retrieval|各チャンクが元のドキュメントのどの位置にあるかを説明する文章を追加|52%|+1%|
|+HyDE|LLMに架空の回答を出力させ、それをクエリとして文書を検索​|58%|+6%|
|+Rerank+マルチクエリ|元のクエリ, rewriteしたクエリ, HyDEで検索した大量の文書の中から、質問との関連性が高い文書のみを抽出​|70%|+12%|

## 🔄 ワークフロー比較

<table>
  <tr>
    <th align="center">🕓 Initial Workflow</th>
    <th align="center">🚀 Updated Workflow</th>
  </tr>
  <tr>
    <td align="center">
      <img src="./asserts/initial-workflow.jpg" alt="Initial Workflow"/>
    </td>
    <td align="center">
      <img src="./asserts/updated-workflow.jpg" alt="Updated Workflow"/>
    </td>
  </tr>
</table>

## 🧪 動作検証

※ Databricks上での実行を想定しています。

1. `faq-chatbot.py`を上から実行する
    - 学校HPのスクレイピングと、DB, ベクトルDBのエンドポイント作成プログラム
    - Run allボタンで全実行
2. `.env`にCohereのAPI keyを追加する
    - Rerankモデルとして、Cohereの`rerank-v3.5`を使用
    - `cp .env.sample .env`をし、COHERE_API_KEYに管理者からもらったAPI keyを配置する
3. `chain_langchain.py`を上から実行する
    - ユーザーからの質問に回答するワークフローを、langchainのLCELを使って実装
    - 動作確認のために、一番下の `# chain.invoke(input_example)`のコメントを外す
    - Run allボタンで全実行

※ `chain_langchain.py`での検証時、エンドポイントがない旨のエラーが出た場合は[./create-vector-db.py](./create-vector-db.py)をすべて実行してエンドポイントを再度作成してください。

## 📝 RAGの定量/定性評価方法

前提: [動作検証](#-動作検証)が完了していること

### 🧠 定性評価方法

- レビューアプリをデプロイし、人間がRAGの回答の評価をすることができます。

※ Databricks Free Edition環境だとデプロイできないかも

#### ▶️ 実行方法

1. `RAG_eval.py`を実行する
2. `deployment_info.review_app_url`で出力されるurlにアクセスし、レビューする

### 📐 定量評価方法

- LLM-as-a-Judgeを使い、RAGの精度評価をします。
  - LLM-as-a-Judgeとは、LLMを使ってRAGを評価する手法です。
- Mosaic AI Agent Evaluationの機能を使って評価します。

#### ✅ 評価基準

以下の4つの観点で評価します。

- Correctness(正確性)：回答は正確か​
- Context sufficiency(文脈充足性)：回答に必要な情報を取得できたか​
- Groundedness(根拠性)：取得した情報に基づいて回答できているか​
- Overall(総合性)：上記三つが全て合格か​

#### ▶️ 実行方法

1. `RAG_eval-LLM-As-A-Judge.py`を実行する
2. `mlflow.evaluate()`で出力された`View run new_eval_run at:`のurlの`Tracing`タブで評価結果を確認できます。
