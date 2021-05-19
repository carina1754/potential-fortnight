import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
data=[]
for i in range(0,50):
    url = "https://search.naver.com/search.naver?&where=news&query=%EC%BD%94%EB%A1%9C%EB%82%98%20%EB%89%B4%EC%8A%A4&sm=tab_pge&sort=1&photo=0&field=0&reporter_article=&pd=0&ds=&de=&docid=&nso=so:dd,p:all,a:all&mynews=0&start={}&refresh_start=0".format(i*10+1)
    headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36'}
    html = requests.get(url, headers = headers)
    soup = BeautifulSoup(html.text, 'html.parser')
    titles_html = soup.select('.news_area > a')
    for i in range(len(titles_html)):
        titles_html[i] = titles_html[i]
        print(titles_html[i]['title'])
    for i in range(len(titles_html)):
        data += [[titles_html[i]['title'],0]]

    dataframe = pd.DataFrame(data,columns=['title', 'level'])
    dataframe.to_csv("sample_submission.csv",header=False,index=False,encoding='utf-8')