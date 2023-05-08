import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils 

from torchvision import transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import *
from nn_model_major import DNN
from sklearn.metrics import mean_squared_error, r2_score

dnn_model = DNN()
dnn_mse = nn.MSELoss()
optimizer = optim.Adam(dnn_model.parameters(), lr = 0.008)

test_pred_list = []
dnn_model.eval()
with torch.no_grad():
    batch_length = len(test_loader)
    avg_mse = 0
    avg_score = 0
    for x_test_batch, y_test_batch in test_loader:
        test_pred = dnn_model(x_test_batch)
        y_test_batch = y_test_batch.reshape(-1, 1)
        mse = dnn_mse(test_pred, y_test_batch)
        score = r2_score(test_pred, y_test_batch)
        avg_mse += mse / batch_length
        avg_score += score / batch_length
        test_pred_list.append(test_pred.numpy())
    print(np.sqrt(avg_mse), avg_score)

# 테스트 데이터 데이터프레임화

import itertools
os.chdir('/Users/choeunsol/Python/PAINS 방중 프로젝트/최종 분석 데이터셋/모델 예측치')

name_nn = data.iloc[y_test_pd.index].loc[:, 'name']

x_test = torch.from_numpy(x_test_pd.values.astype('f'))
y_test = torch.from_numpy(y_test_pd.values.astype('f'))

prediction_dnn = dnn_model(x_test)
prediction_dnn = np.round(prediction_dnn.detach().numpy().astype('f').reshape(-1), 2)

real_y_value = y_test.numpy()

prediction_dnn = scaler_nn.inverse_transform(prediction_dnn.reshape(-1, 1))
real_y_value = scaler_nn.inverse_transform(real_y_value.reshape(-1, 1))

df_dnn = pd.DataFrame({'name' : name_nn, 'NN 예측치' : np.round(prediction_dnn.reshape(-1), 1),
                       '트레이드 가치' : np.round(real_y_value.reshape(-1), 2)})
df_dnn.to_csv('NN model_major level_예측치.csv')