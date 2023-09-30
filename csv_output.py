import os
import numpy as np
import pandas as pd

folder_path = 'results_adspend'
all_files = os.listdir(folder_path)
npy_files = [f for f in all_files if f.endswith('.npy')]
loaded_data = {}

for npy_file in npy_files:
    full_path = os.path.join(folder_path, npy_file)
    data = np.load(full_path)
    
    data = np.array(data).flatten()
    
    key_name = npy_file.replace('.npy', '')
    print(data)
    loaded_data[key_name] = data

df = pd.DataFrame(loaded_data)
df.to_csv("predict.csv", index=False)










