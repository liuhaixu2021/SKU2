import torch
import numpy as np

class Normal_scaler():
    def __init(self):
        self.mean = 0
        self.std = 1


    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data_device) if torch.is_tensor(data) else self.std
        return (data-mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return data * std + mean


class Exp_scaler():
    def __init__(self):
        self.mean = 0

    def fit(self, data):
        self.mean = data.mean(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        centered_data = data - mean
        return np.exp(centered_data)

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        log_data = np.log(data)
        return log_data + mean

class No_scaler():
    def __init__(self):
        self.mean = 0

    def fit(self, data):
        self.mean = data.mean(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        centered_data = data - mean
        return data

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        log_data = np.log(data)
        return data



class MinMax_scaler():
    def __init__(self):
        self.min = 0
        self.max = 1

    def fit(self, data):
        self.min = data.min(0)
        self.max = data.max(0)

    def transform(self, data):
        min_val = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        max_val = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        return (data - min_val) / (max_val - min_val)

    def inverse_transform(self, data):
        min_val = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        max_val = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        return data * (max_val - min_val) + min_val

