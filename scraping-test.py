# Databricks notebook source
# MAGIC %pip freeze

# COMMAND ----------

# openai, httpx, beautifulsoup4 をインストール
%pip install openai httpx beautifulsoup4
dbutils.library.restartPython()  # databricksのpythonを再起動させる

# COMMAND ----------

import os
import openai
import httpx
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re

# COMMAND ----------

load_url = "https://www.tech.ac.jp/sitemap/"
html = httpx.get(load_url)
soup = BeautifulSoup(html.content, "html.parser")

# COMMAND ----------

links = soup.find_all('a')
links_set = set()
for link in links:
    url = link.get('href')
    if "http" not in url and url.startswith("/"):
        url = "https://www.tech.ac.jp" + url  # urlが相対パスになっているため、www.~~を追加
        links_set.add(url)

urls_ls = list(links_set)

# COMMAND ----------

# sitemapから取ってきた、HPのurl一覧
urls_ls

# COMMAND ----------

q_and_a = "https://www.tech.ac.jp/school/faq/"

html = httpx.get(q_and_a)
soup = BeautifulSoup(html.content, "html.parser")

# COMMAND ----------

qa_list = []
qa_selector = soup.select("#page > main > article > #faq01,#faq02,#faq03")

for faq_container in qa_selector:
    faq_item = faq_container.select("div > div > div > ul > li")
    
    for faq in faq_item:
        q = faq.select(".-q > p")[0].text
        a = faq.select(".-a > p")[0].text
        print(q)
        print(a)
        print("-"*40)

        qa_list.append([q, a])

# COMMAND ----------


len(qa_list)

# COMMAND ----------

qa_list

# COMMAND ----------


