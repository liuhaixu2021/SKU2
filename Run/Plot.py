import os
import numpy as np
import matplotlib.pyplot as plt

folder_path = 'results'
all_files = os.listdir(folder_path)
npy_files = [f for f in all_files if f.endswith('.npy')]
loaded_data = {}

for npy_file in npy_files:
    full_path = os.path.join(folder_path, npy_file)
    loaded_data[npy_file] = np.load(full_path)
    

"""
true_ad_90 = loaded_data['true_ad.npy'][:90]
true_ad_90 = true_ad_90.reshape(1, 90, 1)
for key in loaded_data:
    if key != 'true_ad.npy':
        loaded_data[key] = np.concatenate((true_ad_90, loaded_data[key]), axis=1)
"""

for model, values in loaded_data.items():
    plt.plot(values[0], label=model.split('.')[0])

plt.legend()
plt.title('Model Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Values')

output_path = os.path.join("model_predictions.png")
plt.savefig(output_path)




