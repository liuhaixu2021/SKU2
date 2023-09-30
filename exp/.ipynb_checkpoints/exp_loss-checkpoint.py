import torch.nn as nn
import torch
import numpy as np


def dtw_distance(y_pred, y_true):
    """
    DTW = {}
    for i in range(len(y_pred)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(y_true)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(y_pred)):
        for j in range(len(y_true)):
            dist = (y_pred[i] - y_true[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(y_pred)-1, len(y_true)-1])
    """
    
    return 0


def compute_increments(sequence):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.cpu().detach()

    diffs = np.diff(sequence.numpy())
    increments = (diffs > 0).astype(int)
    return increments

def hamming_distance(y_pred, y_true):
    return np.sum(y_pred != y_true)


def sequence_increments_distance(y_pred, y_true):
    inc_seq1 = compute_increments(y_pred)
    inc_seq2 = compute_increments(y_true)

    return hamming_distance(inc_seq1, inc_seq2)


def MSE(y_pred, y_true):
    mse_loss = nn.MSELoss()
    mse_value = mse_loss(y_pred, y_true)
    return mse_value

def MAE(y_pred, y_true):
    mae_loss = nn.L1Loss()
    mae_value = mae_loss(y_pred, y_true)
    return mae_value

def MAPE(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))

def R2(y_pred, y_true):
    sst = torch.sum((y_true - y_true.mean()) ** 2)
    ssr = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ssr / sst)
    return r2





class CustomLoss(nn.Module):
    def __init__(self, args):
        super(CustomLoss, self).__init__()
        self.theta = args.para_mes
        self.alpha = args.para_var
        self.beta = args.para_dtw
        self.gamma = args.para_sid
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        mse_loss = self.mse(predictions, targets)
        variance_penalty = predictions.var()
        batch_size, seq_len, temp = predictions.shape

        dtw_loss = 0
        for i in range(batch_size):
            dtw_loss += dtw_distance(predictions[i].cpu().detach().numpy(), targets[i].cpu().detach().numpy())
        distance_penalty = torch.tensor(dtw_loss, requires_grad=True).to(predictions.device) / batch_size

        hamming_loss = sequence_increments_distance(predictions, targets)
        return self.theta * mse_loss + self.alpha * variance_penalty + self.gamma * hamming_loss
        """
        mse_loss = self.mse(predictions, targets)
        variance_penalty = predictions.var()
        return self.theta * mse_loss




