import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

url = "https://news.naver.com/main/factcheck/main.nhn?page=1&section=%C4%DA%B7%CE%B3%AA+%B9%D9%C0%CC%B7%AF%BD%BA"
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

dataframe = pd.DataFrame(data,columns=['title', 'level'])
dataframe.to_csv("1.csv",header=False,index=False,encoding='utf-8')

print(data)

for i in range(2,32):
    url = "https://news.naver.com/main/factcheck/main.nhn?page={}&section=%C4%DA%B7%CE%B3%AA+%B9%D9%C0%CC%B7%AF%BD%BA".format(i)
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