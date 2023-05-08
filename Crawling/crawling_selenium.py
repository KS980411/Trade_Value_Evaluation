## 라이브러리 임포트

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

import bs4
from bs4 import BeautifulSoup
import selenium
from selenium import webdriver

driver = webdriver.Chrome("/Users/choeunsol/chromedriver.exe")
driver.close()

# 동적 홈페이지 크롤링 by Selenium

dr = webdriver.Chrome()
dr.get("https://www.baseballtradevalues.com/players/")

from selenium.webdriver.common.by import By
import requests

text_list = []
name_list = []

## 페이지 넘기기(버튼 자동화) 

for _ in range(78):
    button_start = dr.find_elements(By.XPATH, '//*[@id="allPlayers_next"]')[0]
    table = dr.find_elements(By.XPATH, '//*[@id="allPlayers"]')[0]
    for i in range(50):
        tr = table.find_elements(By.TAG_NAME, "tbody")[0].find_elements(By.TAG_NAME, "tr")[i]
        name = tr.find_elements(By.TAG_NAME, 'td')[0].text
        text_list.append(text)
        name_list.append(name)
    button_start.click()

for _ in range(78):
    button_init = dr.find_elements(By.XPATH, '//*[@id="allPlayers_previous')[0]
    button_init.click()

## 웹 크롤링

name_list = []
age_list = []
level_list = []
position_list = []
avail_list = []
control_years_list = []
afv_list = []
salary_list = []
surplus_list = []
median_tv_list = []

for _ in range(78): # 에러창 안뜨려면 77로 하고 마지막 페이지는 따로 반복문을 만들어야 한다.
    button_start = dr.find_elements(By.XPATH, '//*[@id="allPlayers_next"]')[0]
    table = dr.find_elements(By.XPATH, '//*[@id="allPlayers"]')[0]
    for i in range(50):
        tr = table.find_elements(By.TAG_NAME, "tbody")[0].find_elements(By.TAG_NAME, "tr")[i]
        name = tr.find_elements(By.TAG_NAME, 'td')[0].text
        age = tr.find_elements(By.TAG_NAME, 'td')[1].text
        level = tr.find_elements(By.TAG_NAME, 'td')[2].text
        position = tr.find_elements(By.TAG_NAME, 'td')[3].text
        avail = tr.find_elements(By.TAG_NAME, 'td')[5].text
        control_years = tr.find_elements(By.TAG_NAME, 'td')[6].text
        afv = tr.find_elements(By.TAG_NAME, 'td')[7].text
        salary = tr.find_elements(By.TAG_NAME, 'td')[8].text
        surplus = tr.find_elements(By.TAG_NAME, 'td')[9].text
        median_tv = tr.find_elements(By.TAG_NAME, 'td')[11].text
        
        text_list.append(text)
        name_list.append(name)
        age_list.append(age)
        level_list.append(level)
        position_list.append(position)
        avail_list.append(avail)
        control_years_list.append(control_years)
        afv_list.append(afv)
        salary_list.append(salary)
        surplus_list.append(surplus)
        median_tv_list.append(median_tv)
        
    button_start.click()

# 이동한 페이지 처음 페이지로 되돌려놓기

for _ in range(78):
    button_init = dr.find_elements(By.XPATH, '//*[@id="allPlayers_previous"]')[0]
    button_init.click()

# Dataframe화

dict_data = {"name" : name_list, "age" : age_list, "level" : level_list, "pos" : position_list, "availability" : avail_list, "control years" : control_years_list, "field value" : afv_list, "salary" : salary_list, "surplus" : surplus_list, "median trade value" : median_tv_list}
data = pd.DataFrame(dict_data)
data