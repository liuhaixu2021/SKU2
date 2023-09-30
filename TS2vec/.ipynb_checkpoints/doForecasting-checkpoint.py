from ts2vec import TS2Vec
import datautils
from forecasting import eval_forecasting
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch, gc
import numpy as np  # Added for reshaping

gc.collect()
torch.cuda.empty_cache()

# Load data
data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv('SKU2', univar=False)

assert data.ndim in [2, 3], f"Expected data to be 2D or 3D, got shape {data.shape}"

if data.ndim == 2:
    data = np.expand_dims(data, axis=0)

assert data.ndim == 3, f"Expected data to be 3D, got shape {data.shape}"

# Check data dimensions
if data.ndim == 3:
    data = data.squeeze(0)  # Remove the first dimension if it's 3D
elif data.ndim != 2:
    raise ValueError(f"Unexpected number of dimensions: {data.ndim}")

# Convert the loaded data to a DataFrame if it's not
if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)

# Convert all columns to numeric, setting unconvertible values to NaN
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows containing NaN values
data.dropna(inplace=True)

assert not data.isna().any().any(), "DataFrame still contains NaN values after dropping them"


# Check if DataFrame is empty
if data.empty:
    raise ValueError("DataFrame is empty after dropping NaN rows.")

# Update slices in case rows have been removed
start = max(train_slice.start if train_slice.start else 0, 0)
stop = min(train_slice.stop if train_slice.stop else len(data), len(data))
train_slice = slice(start, stop)

# Check if slice contains data
if len(data[train_slice]) == 0:
    raise ValueError("train_slice does not select any samples.")

original_shape = data.loc[train_slice].values.shape

# Calculate the number of samples and number of features
n_samples, n_features = original_shape

# Reshape train_data to have three dimensions (1, n_samples, n_features)
train_data = np.reshape(data.loc[train_slice].values, (1, n_samples, n_features))

#data = np.expand_dims(data, axis=0)

assert train_data.ndim == 3, f"Expected train_data to be 3D, got shape {train_data.shape}"

# Additional check for NaN in train_data
assert not np.isnan(train_data).any(), "train_data contains NaN values"

# Check if train_data shape is as expected
if train_data.ndim != 3:
    raise ValueError(f"train_data should be 3D. Got shape {train_data.shape}")

    


# Model training
model = TS2Vec(
    input_dims=train_data.shape[-1],
    device='cuda',
    batch_size=2,
    lr=0.001,
    max_train_length=3000,
    output_dims=320,
)

if train_data.shape[1] == 0:
    raise ValueError("train_data is empty. Check your data loading and slicing steps.")
    
loss_log = model.fit(
    train_data,
    verbose=True
)

out, eval_res = eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
print(eval_res)
print(out)
