import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
change_value_dict = {'전혀 사실 아님' : 0,'대체로 사실 아님' : 0.25,'절반의 사실' : 0.5,'대체로 사실' : 0.75, '사실' : 1}

url = "https://news.naver.com/main/factcheck/main.nhn?section=%C4%DA%B7%CE%B3%AA%B9%E9%BD%C5"
headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36'}
html = requests.get(url, headers = headers)
soup = BeautifulSoup(html.text, 'html.parser')
titles_html = soup.select('.section > .talk_area >.info_area > .txt > a')
titles_level = soup.select('.bx .blind')
data = []
for i in range(len(titles_html)):
    titles_level[i] = titles_level[i].text
    titles_html[i] = titles_html[i].text
    data += [[titles_html[i],titles_level[i]]]

data = data.replace({'level' : change_value_dict})
dataframe = pd.DataFrame(data,columns=['title', 'level'])
dataframe.to_csv("1.csv",header=False,index=False,encoding='utf-8')

print(data)

for i in range(2,32):
    url = "https://news.naver.com/main/factcheck/main.nhn?page={}&section=%C4%DA%B7%CE%B3%AA%B9%E9%BD%C5".format(i)
    headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36'}
    html = requests.get(url, headers = headers)
    soup = BeautifulSoup(html.text, 'html.parser')
    titles_html = soup.select('.section > .talk_area >.info_area > .txt > a')    
    titles_level = soup.select('.bx .blind')
    for i in range(len(titles_level)):
        titles_level[i] = titles_level[i].text
    for i in range(len(titles_html)):
        titles_html[i] = titles_html[i].text
    for i in range(len(titles_html)):
        data += [[titles_html[i],titles_level[i]]]
    
    dataframe = pd.DataFrame(data,columns=['title', 'level'])
    dataframe.to_csv("1.csv",header=False,index=False,encoding='utf-8')