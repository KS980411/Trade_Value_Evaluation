import bs4
import numpy as np
import requests
from bs4 import BeautifulSoup as bs

# url 리스트 제작 - url이 번호만 다른 것을 이용

url_list = []
number_list = np.arange(0, 1140, 30)
number_list # url 번호 리스트 확인

# 타자 데이터

response = []

for url in url_list:
    response_text = requests.get(url)
    response_text.raise_for_status()
    soup = bs(response_text.text, 'html.parser')
    response.append(soup)

p_list = []

for i in range(38):
    html_table = response[i].find_all('table')
    p = parser.make2d(html_table[2])
    p_list.append(p)

# 데이터 프레임화

df_statiz_war = pd.DataFrame(p_list)
df_statiz_war = df_statiz_war.iloc[1:len(df_statiz_war)]

df_statiz_war.to_excel('20-22 스탯티즈 타자 WAR.xlsx', header = None)

