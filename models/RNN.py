import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_dim = 1
        self.hidden_dim = configs.hidden_dim
        self.output_dim = 1 * configs.pred_len
        self.num_layers = configs.num_layers

        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out[:, -1, :])
        return out.unsqueeze(2)