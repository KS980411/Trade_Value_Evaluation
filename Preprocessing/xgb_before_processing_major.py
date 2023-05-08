import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scaling import scaled_into_trainset

# 간단한 전처리

os.chdir('/Users/choeunsol/Python/PAINS 방중 프로젝트/최종 분석 데이터셋/인풋 데이터')
data = pd.read_csv('MLB 메이저레벨 분석용 데이터셋.csv')
data.head()

setfordata = data.loc[:, ['age', 'field.value', 'pos', 'control.years', 'fwar_19', 'fwar_21', 'fwar_22', 'options', 'salary']]

## 라벨 인코딩 : 포지션 변수를 이산형 변수로 설정

pos_dict = {"1B" : 1, "2B" : 2, "3B" : 3, "C" : 4, "DH" : 5, "OF" : 6, "RP" : 7, "SP" : 8, "SS" : 9, "UTIL" : 10}
afterdata = setfordata['pos'].apply(lambda x : pos_dict[x])
setfordata['pos'] = afterdata

## X와 Y로 나누기

dataset = pd.get_dummies(setfordata).drop(['options_Yes', 'options_No'], axis = 1)
Y = dataset.loc[:, 'field.value']
X_col = list(dataset.columns.difference(['field.value']))
X = dataset[X_col]

## 정규화

import random
random.seed(30)

from sklearn.preprocessing import RobustScaler,StandardScaler

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 1/3, random_state = 0)

name = data.iloc[test_y.index].loc[:, 'name']

# mlb_fv_scaler = StandardScaler()
mlb_fv_scaler = RobustScaler()
mlb_fv_scaler.fit(train_y.values.reshape(-1,1))

mlb_salary_scaler = RobustScaler()
mlb_salary_scaler.fit(train_x.loc[:,'salary'].values.reshape(-1,1))

trainx_scalingData = train_x.loc[:, ['salary']]
testx_scalingData = test_x.loc[:, ['salary']]
validx_scalingData = valid_x.loc[:, ['salary']]

xscaler = scaled_into_trainset(RobustScaler())

scaled_train_x, scaled_test_x, scaled_valid_x = xscaler.x_scaled(trainx_scalingData, testx_scalingData, validx_scalingData)
train_x.loc[:, ['salary']] = scaled_train_x
test_x.loc[:, ['salary']] = scaled_test_x
valid_x.loc[:, ['salary']] = scaled_valid_x

yscaler = scaled_into_trainset(StandardScaler())
train_y, test_y, valid_y = yscaler.y_scaled(train_y, test_y, valid_y)

