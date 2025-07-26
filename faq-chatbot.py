# Databricks notebook source
# MAGIC %pip freeze

# COMMAND ----------

# MAGIC %pip install openai httpx beautifulsoup4

# COMMAND ----------

# MAGIC %pip install databricks-langchain=0.1.1
# MAGIC %pip install mlflow lxml==4.9.3 transformers==4.30.2 databricks-vectorsearch==0.22 databricks-sdk==0.28.0 databricks-feature-store==0.17.0 langchain==0.2.11 langchain_core==0.2.23 langchain-community==0.2.9 databricks-agents
# MAGIC %pip install dspy-ai -U

# COMMAND ----------

# databricksのpythonを再起動させる
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %pip list

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scraping TECH.C. Info

# COMMAND ----------

import os
import openai
import httpx
from bs4.element import Tag
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re

# COMMAND ----------

# 情報収集に使えそうなリンク集
# https://www.tech.ac.jp/sitemap/
# https://www.tech.ac.jp/work_books/
# https://www.tech.ac.jp/blog/

# まだ使えていないページ
# https://content.tech.ac.jp/campuslife/

# PDFいっぱいあるページ
# https://www.tech.ac.jp/info/

# 使えそうなPDF
# https://www.tech.ac.jp/pdf/info/2-1-1.pdf
# https://www.tech.ac.jp/pdf/info/2-3-1.pdf
# https://www.tech.ac.jp/pdf/info/7-2.pdf
# https://www.tech.ac.jp/pdf/info/9-2.pdf

# COMMAND ----------

from httpx import Timeout
import time

def get_soup(url):
    with httpx.Client() as client:
        # ReadTimeoutが起きることがあるので、情報取得のタイムアウトを10sにする
        timeout = Timeout(5.0, read=10.0)
        html = client.get(url, timeout=timeout, follow_redirects=True)
    if html.status_code != 200:
        print(f"Failed to get {url}")
        html.raise_for_status()
    return BeautifulSoup(html.content, "html.parser")

# COMMAND ----------

sitemap_url = "https://www.tech.ac.jp/sitemap/"
sitemap_soup = get_soup(sitemap_url)
work_books_url = "https://www.tech.ac.jp/work_books/"
work_books_soup = get_soup(work_books_url)
blog_url = "https://www.tech.ac.jp/blog/"
blog_soup = get_soup(blog_url)

# COMMAND ----------

def extract_links(soup) -> set:
    links = soup.find_all('a')
    links_set = set()
    for link in links:
        url = link.get('href')
        if "http" not in url and url.startswith("/"):
            url = "https://www.tech.ac.jp" + url  # urlが相対パスになっているため、www.~~を追加
            links_set.add(url)

    return links_set

# COMMAND ----------

sitemap_urls = extract_links(sitemap_soup)
work_books_urls = extract_links(work_books_soup)
blog_urls = extract_links(blog_soup)

urls_ls = list(sitemap_urls | work_books_urls | blog_urls)

# COMMAND ----------

# pdfはいったん退避
for url in urls_ls:
  if url.endswith('pdf'):
    print(url)  # https://www.tech.ac.jp/pdf/admission/how-to.pdf
    urls_ls.remove(url)


# COMMAND ----------

# リンク集は省く
urls_ls.remove(sitemap_url)
urls_ls.remove(work_books_url)
urls_ls.remove(blog_url)

# 学生作品紹介ページ
student_works_url = 'https://www.tech.ac.jp/gallery/'
urls_ls.remove(student_works_url)

# 韓国語、英語、中国語の紹介ページ
lang_urls = [
  'https://www.tech.ac.jp/visitor/language/ko/about/',
  'https://www.tech.ac.jp/visitor/language/en/about/',
  'https://www.tech.ac.jp/visitor/language/ch/about/'
]
for url in lang_urls:
  urls_ls.remove(url)

# 変動するページは一旦やらないようにする
# 理由は、RAGが回答する時期にこのオープンキャンパスがやっているとは限らないから
# もしやるとしたら、イベント用のテーブルを別で作って、月一で更新かけるとかが有効かも
event_urls = [
  'https://www.tech.ac.jp/opencampus/',
  'https://www.tech.ac.jp/opencampus/program/special-event/',
  'https://www.tech.ac.jp/opencampus/program/experience-lesson/',
  'https://www.tech.ac.jp/opencampus/program/exam/',
  'https://www.tech.ac.jp/opencampus/program/school-briefing/',
]
for url in event_urls:
  urls_ls.remove(url)


# COMMAND ----------

# 取得したHPのurl一覧
for url in urls_ls:
    print(url)

# COMMAND ----------

len(urls_ls)

# COMMAND ----------

from typing import Optional

def extract_page_different(soup) -> Tag:
  # main tagがない
  article_html = soup.select("#page > .l-contents > .l-main > article")[0]
  course_info = article_html.select(".p-different_course")
  opencampus_info = article_html.select(".p-different_opencampus")
  article_html.select(".p-different_course")[0].decompose()
  article_html.select(".p-different_opencampus")[0].decompose()
  return article_html

def extract_page_strengths(soup) -> Tag:
  # main tagがない
  return soup.select("#page > .l-contents > .l-main > article")[0]

def extract_page_myschool(soup) -> str:
  # article tagがない
  opencampus_leading = soup.select("#page > main > .p-opencampus_leading")[0]
  myschool_point = soup.select("#page > main > .p-myschool_point")[0]
  return str(opencampus_leading) + str(myschool_point)

def extract_page_webopencampus(soup) -> str:  
  # article tagがない
  opencampus_leading = soup.select("#page > main > .p-opencampus_leading")[0]
  unique_info = soup.select("#page > main > .c-common_section")[0]
  web_opencampus_step = soup.select("#page > main > .c-common_section")[0]
  return str(opencampus_leading) + str(web_opencampus_step)

def extract_article_html(url) -> Optional[str]:
  soup = get_soup(url)
  
  try:
    article_html = soup.select("#page > main > article")[0]
    remove_tag_selectors = [
      ".c-lower_links",# リンク集
      ".p-course_opencampus",# OpenCampus情報
      "#opencampus",# OpenCampus情報
      ".p-course_major",# 専攻一覧
      ".p-world_links",# 専攻リンク
      ".c-cta01_sm",# 資料請求とopemcampus
      ".p-work_books_article__body > .p-work_books__opencampus",# workbook内のopemcampus情報
      ".c-admission_cta" # パンフレット
    ]
    combined_selector = ", ".join(remove_tag_selectors)
    for tag in article_html.select(combined_selector):
      tag.decompose()

  except:
    # 個別処理
    print('個別処理 url:', url)
    if url == 'https://www.tech.ac.jp/features/different/':
      article_html = extract_page_different(soup)
    elif url == url == 'https://www.tech.ac.jp/features/strengths/':
      article_html = extract_page_strengths(soup)
    elif url == 'https://www.tech.ac.jp/myschool/':
      article_html = extract_page_myschool(soup)
    elif url == 'https://www.tech.ac.jp/web_opencampus/':
      article_html = extract_page_webopencampus(soup)
    else:
      print('error in url:', url)
      return None

  return str(article_html)

# COMMAND ----------

url_html_pairs = []

import time
from tqdm import tqdm
print('Processing...')
for url in tqdm(urls_ls, desc='Extracting HTML'):
  article_html = extract_article_html(url)
  if article_html is not None:
    url_html_pairs.append({
      'url': url,
      'text': article_html,
    })
  else:
    print('error in url:', url)
  time.sleep(0.5)
print('All done!')

# COMMAND ----------

len(url_html_pairs)

# COMMAND ----------

url_html_pairs[0]

# COMMAND ----------

# 参考
# https://qiita.com/taka_yayoi/items/f174599e4721e51e9e1d
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, OpenAIGPTTokenizer

max_chunk_size = 512

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

# Split on H2で分割しますが、あまり小さすぎないように小さなh2チャンクはマージします
def split_html_on_h2(html, min_chunk_size = 20, max_chunk_size=512):
    if not html:
        return []
    h2_chunks = html_splitter.split_text(html)
    page_contents = "".join([c.page_content for c in h2_chunks])
    chunks = []
    previous_chunk = ""
    results = []
    # チャンクを結合し、h2の前にテキストを追加することでチャンクを結合し、小さすぎる文書を回避します
    for c in h2_chunks:
        # h2の結合 (注意: 重複したh2を回避するために以前のチャンクを削除することもできます)
        current_h2 = c.metadata.get('header2', "")  # 現在のH2
        content = current_h2 + "\n" + c.page_content
        if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size/2:
            previous_chunk += content + "\n"
        else:
            split_chunks = text_splitter.split_text(previous_chunk.strip())
            for chunk in split_chunks:
                results.append({
                    "content": chunk,
                    "page_contents": page_contents
                })
            previous_chunk = content + "\n"
    if previous_chunk:
        split_chunks = text_splitter.split_text(previous_chunk.strip())
        for chunk in split_chunks:
            results.append({
                "content": chunk,
                "page_contents": page_contents
            })
    # 小さすぎるチャンクの破棄
    return [r for r in results if len(tokenizer.encode(r["content"])) > min_chunk_size]
 

# COMMAND ----------

html = url_html_pairs[0]['text']
split_html = split_html_on_h2(html)

# COMMAND ----------

split_html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Table

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import mlflow
from mlflow import MlflowClient

# COMMAND ----------

# sparkですべてのドキュメントのチャンクを作成するためのユーザー定義関数(UDF)を作成
@pandas_udf("array<struct<content:string, page_contents:string>>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)
    

# COMMAND ----------

def use_and_create_db(catalog, dbName, cloud_storage_path = None):
  print(f"CREATE CATALOG IF NOT EXISTS {catalog};")
  sql(f"CREATE CATALOG IF NOT EXISTS {catalog};")
  print(f"USE CATALOG `{catalog}`")
  spark.sql(f"USE CATALOG `{catalog}`")
  spark.sql(f"""create database if not exists `{dbName}` """)

assert catalog not in ['hive_metastore', 'spark_catalog']
#If the catalog is defined, we force it to the given value and throw exception if not.
if len(catalog) > 0:
  current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
  if current_catalog != catalog:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if catalog not in catalogs:
      # spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
      if catalog == 'dbdemos':
        spark.sql(f"ALTER CATALOG {catalog} OWNER TO `account users`")
  use_and_create_db(catalog, dbName)

# COMMAND ----------

[r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]

# COMMAND ----------

def use_catalog_and_create_schema():
    # sql(f"CREATE CATALOG IF NOT EXISTS {catalog};")
    sql(f"USE CATALOG {catalog};")
    sql(f"CREATE SCHEMA IF NOT EXISTS {dbName};")
    sql(f"USE SCHEMA {dbName};")
    sql(f"CREATE VOLUME IF NOT EXISTS {volume};")

use_catalog_and_create_schema()

# COMMAND ----------

# すでに同名のテーブルが存在する場合は削除
html_raw_data_table_name = f'html_{raw_data_table_name}'

# COMMAND ----------

sql(f"drop table if exists {html_raw_data_table_name}")


spark.createDataFrame(url_html_pairs).write.mode('overwrite').saveAsTable(html_raw_data_table_name)

display(spark.table(html_raw_data_table_name))

# COMMAND ----------

sql(f"drop table if exists {raw_data_table_name}")

# すべてのドキュメントチャンクを保存する
(spark.table(html_raw_data_table_name)
      .filter('text is not null')
      .withColumn('split_content', F.explode(parse_and_split('text')))
      .selectExpr("split_content.content as content", "split_content.page_contents as page_contents", 'url')
      .write.saveAsTable(raw_data_table_name))

display(spark.table(raw_data_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Contextual-Retrieval

# COMMAND ----------

use_catalog_and_create_schema()

# COMMAND ----------

# MAGIC %run ./approaches/embedding/contextual_retrieval

# COMMAND ----------

# dictにする

raw_data_table_df = spark.table(raw_data_table_name).toPandas()
raw_data_dict = raw_data_table_df.to_dict(orient='records')

len(raw_data_dict)

# COMMAND ----------

import time 
from tqdm import tqdm
processes_list = []
failed_list = []

for doc in tqdm(raw_data_dict):
    try:
        # たまにmlflowでkeyerror:'content'という謎のエラーが出るので、ここでtry-catchしておく
        processed_content = process_and_annotate_document(doc['content'], doc['page_contents'])
    except Exception as e:
        failed_list.append(doc)
        time.sleep(5)
        processed_content = process_and_annotate_document(doc['content'], doc['page_contents'])

    processed_raw_data_dict = {
        'content': processed_content,
        'page_contents': doc['page_contents'],
        'url': doc['url'],
    }
    processes_list.append(processed_raw_data_dict)
    time.sleep(0.5)

# COMMAND ----------

failed_list

# COMMAND ----------

len(processes_list)

# COMMAND ----------

spark.createDataFrame(processes_list).write.mode('overwrite').saveAsTable(raw_data_table_name)
display(spark.table(raw_data_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add custom data

# COMMAND ----------

# f'{raw_data_table_name}_original'として、raw_data_table_nameを保存
spark.table(raw_data_table_name).write.mode('overwrite').saveAsTable(f'{raw_data_table_name}_original')

# COMMAND ----------

raw_data_table_df = spark.table(raw_data_table_name).toPandas()

add_data_df = pd.read_csv(f'./add-data.csv')
# urlとpage_contentsを""にする
add_data_df = add_data_df.fillna("")

raw_data_table_merged_df = pd.concat([raw_data_table_df, add_data_df], ignore_index=True)
raw_data_dict = raw_data_table_merged_df.to_dict(orient='records')

spark.createDataFrame(raw_data_dict).write.mode('overwrite').saveAsTable(raw_data_table_name)
display(spark.table(raw_data_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Embedding table

# COMMAND ----------

use_catalog_and_create_schema()

# COMMAND ----------

sql(f"DROP TABLE IF EXISTS {embed_table_name};")

sql(f"""
--インデックスを作成するには、テーブルのChange Data Feedを有効にします
CREATE TABLE IF NOT EXISTS {embed_table_name} (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY,
  url STRING,
  content STRING,
  page_contents STRING
) TBLPROPERTIES (delta.enableChangeDataFeed = true); 
""")

spark.table(raw_data_table_name).write.mode('overwrite').saveAsTable(embed_table_name)

display(spark.table(embed_table_name))

# COMMAND ----------

import time

def index_exists(vsc, endpoint_name, index_full_name):
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get('status').get('ready', False)
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False
  

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

from databricks.vector_search.client import VectorSearchClient

# エンドポイントは自分で立ち上げる
# エンドポイントは毎回立ち上げるもの -> ずっと起動していると、お金がかかっちゃう
# エンドポイントを落とすと、紐づいているベクトルインデックスが全て削除されちゃう
    # 毎度起動する際は、ベクトルインデックスも作らなければならない
vsc = VectorSearchClient()

if VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
import time

#インデックスの元となるテーブル
source_table_fullname = f"{catalog}.{db}.{embed_table_name}"

#インデックスを格納する場所
vs_index_fullname = f"{catalog}.{db}.{embed_table_name}_vs_index"

#すでに同名のインデックスが存在すれば削除
if index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Deleting index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.delete_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  while True:
    if index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
      time.sleep(1)
      print(".")
    else:      
      break

#インデックスを新規作成
print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
vsc.create_delta_sync_index(
  endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
  index_name=vs_index_fullname,
  pipeline_type="TRIGGERED",
  source_table_name=source_table_fullname,
  primary_key="id",
  embedding_source_column="content",
  embedding_model_endpoint_name=embedding_endpoint_name
)

#インデックスの準備ができ、すべてエンベッディングが作成され、インデックスが作成されるのを待ちましょう。
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

#同期をトリガーして、テーブルに保存された新しいデータでベクターサーチのコンテンツを更新
vs_index = vsc.get_index(
  VECTOR_SEARCH_ENDPOINT_NAME, 
  vs_index_fullname)

try:
    vs_index.sync()
except Exception as e:
    import time
    time.sleep(10)
    vs_index.sync()  # なぜかエラー出るが、sync()を2回実行するとエラーが出なくなる


# COMMAND ----------

# インデックスへの参照を取|得
vs_index = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# 英語用のモデルを使っているため、回答が良いものではない
# embeddingモデルを日本語用にファインチューニングさせるのもいいかも
  # お家GPUで学習が可能 -> databricksにアップロードする

results = vs_index.similarity_search(
  query_text="授業時間は一コマどのくらいですか？",
  columns=["url", "content"],
  num_results=10,  # 上位三つの結果を返す
)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# RAGチェーンとして呼び方を統一する！

import yaml
import mlflow

rag_chain_config = {
      "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
      "vector_search_index_name": f"{catalog}.{dbName}.{embed_table_name}_vs_index",
      "llm_endpoint_name": instruct_endpoint_name,
      "llm_mini_endpoint_name": instruct_mini_endpoint_name
}
config_file_name = 'rag_chain_config.yaml'
try:
    with open(config_file_name, 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')

# COMMAND ----------

import os

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_HOST"] = API_ROOT
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"] = API_TOKEN

# COMMAND ----------


