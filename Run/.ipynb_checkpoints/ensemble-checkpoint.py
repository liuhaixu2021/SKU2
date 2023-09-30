import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
import pandas as pd

folder_path = 'results'
all_files = os.listdir(folder_path)
npy_files = [f for f in all_files if f.endswith('.npy')]
data_dict = {}

for npy_file in npy_files:
    full_path = os.path.join(folder_path, npy_file)
    data_dict[npy_file] = np.load(full_path)
    
print(data_dict)
    
y = data_dict['true_ad.npy'][-7:]

X_list = [data_dict[key][0, -7:, 0] for key in data_dict if key in ['PatchTST.npy', 'DLinear.npy', 'NLinear.npy']]
print(X_list)
X = np.column_stack(X_list)

model = LinearRegression()
model.fit(X, y)
#lasso = Lasso(alpha=1.0)
#lasso.fit(X, y)

coefficients = {key: coef for key, coef in zip([k for k in data_dict.keys() if k != 'true_ad.npy'], model.coef_)}
intercept = model.intercept_

coeff_df = pd.DataFrame([coefficients])
coeff_df.to_csv('coefficients.csv', index=False)






