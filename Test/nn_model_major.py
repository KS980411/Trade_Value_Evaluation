# 라이브러리 임포트
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils 
import os, sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scaling import scaled_into_trainset
from torchvision import transforms

# 데이터 불러오기

os.chdir('/Users/choeunsol/Python/PAINS 방중 프로젝트/최종 분석 데이터셋/인풋 데이터')
data = pd.read_csv('MLB 메이저레벨 분석용 데이터셋.csv')
data.head()

# 데이터 정규화

y_pd = data_pre_dnn.loc[:,'field.value']
X_pd_c = list(data_pre_dnn.columns.difference(['field.value', 'options_No', 'options_Yes']))
X_pd = data_pre_dnn[X_pd_c]

x_train_pd, x_test_pd, y_train_pd, y_test_pd = train_test_split(X_pd, y_pd, test_size=0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler

scaler_nn = StandardScaler()

# x 스케일링
scaling_nn_x_train = x_train_pd.loc[:, ['salary']]
scaling_nn_x_test = x_test_pd.loc[:, ['salary']]

scaler_nn.fit(scaling_nn_x_train)
scaling_nn_x_test = x_test_pd.loc[:, ['salary']]

trainscaled_x = scaler_nn.transform(scaling_nn_x_train)
testscaled_x = scaler_nn.transform(scaling_nn_x_test)

# y 스케일링
train_y_array = np.array(y_train_pd)
train_y_array = train_y_array.reshape(-1, 1)

test_y_array = np.array(y_test_pd)
test_y_array = test_y_array.reshape(-1, 1)

scaler = scaler_nn.fit(train_y_array)

trainscaled_y = scaler.transform(train_y_array)
testscaled_y = scaler.transform(test_y_array)

# 데이터에 덧씌우기
x_train_pd.loc[:, 'salary'], x_test_pd.loc[:, 'salary'] = trainscaled_x, testscaled_x
y_train_pd[:], y_test_pd[:] = trainscaled_y.reshape(-1), testscaled_y.reshape(-1)


## 모델링

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(nn.Linear(16, 300, bias=True),
                                          nn.ELU(),
                                          nn.Linear(300, 100, bias = True),
                                          nn.ELU(),
                                          nn.Linear(100, 50, bias = True),
                                          nn.ELU(),
                                          nn.Linear(50, 30, bias = True),
                                          nn.ELU(),
                                          nn.Linear(30, 20, bias = True),
                                          nn.ELU(),
                                          nn.Linear(20, 1, bias = True))
        
    def forward(self, x):
        out = self.linear_stack(x)
        return out

