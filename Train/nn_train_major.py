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


dnn_model = DNN()
dnn_mse = nn.MSELoss()
optimizer = optim.Adam(dnn_model.parameters(), lr = 0.008)


total_loss=0
dnn_model.train()
for epoch in tqdm(range(100)):
    avg_loss = 0
    total_loss=0

    for step, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        predict = dnn_model(x_batch)
        y_batch = y_batch.reshape(-1, 1)
        loss = dnn_mse(predict, y_batch)
        loss.backward()
        optimizer.step()
        batch_length = len(y_batch)

        avg_loss = loss.item() / batch_length
        total_loss+=avg_loss
    print(total_loss/len(train_loader))