# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:45:24 2022

@author: AA
"""
import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    if n < pred_len:
        pred_len = n  # Or other handling logic
    features = features[:, :-pred_len]
    labels = np.stack([data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), labels.reshape(-1, labels.shape[2]*labels.shape[3])


def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    print(f"Data shape: {data.shape}")
    print(f"Train slice: {train_slice}, Valid slice: {valid_slice}, Test slice: {test_slice}")
    
    padding = 200
    t = time.time()
    
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
        
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    print(f"All repr shape: {all_repr.shape}")

    ts2vec_infer_time = time.time() - t
    
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
    print(f"Train repr shape: {train_repr.shape}, Valid repr shape: {valid_repr.shape}, Test repr shape: {test_repr.shape}")
    print(f"Train data shape: {train_data.shape}, Valid data shape: {valid_data.shape}, Test data shape: {test_data.shape}")
    
    out_log = {}
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    
    for pred_len in pred_lens:
        # Generate features and labels for training, validation, and testing
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        print(f"Test features shape: {test_features.shape}, Test labels shape: {test_labels.shape}")
        if test_features.shape[0] == 0 or test_labels.shape[0] == 0:
            raise ValueError("test_features or test_labels is empty. Check the slicing and data preparation steps.")
        # Fit Ridge model
        t = time.time()
        lr = fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        
        # Make predictions
        t = time.time()
        if test_features.shape[0] == 0:
            raise ValueError("test_features is empty. Check the slicing and data preparation steps.")
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t
        
        # Reshape and inverse transform predictions
        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)
        
        test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, test_pred.shape[-1])).reshape(test_pred.shape)
        test_labels_inv = scaler.inverse_transform(test_labels.reshape(-1, test_labels.shape[-1])).reshape(test_labels.shape)
        
        # Calculate metrics
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        
        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    
    return out_log, eval_res



def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    
    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr