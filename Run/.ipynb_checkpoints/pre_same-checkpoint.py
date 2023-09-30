import argparse
import os
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')


parser.add_argument('--root_path', type=str, default='./dataset', help='dataset type')
parser.add_argument('--data_path', type=str, required=True, default='SKU_1.csv', help='data file')
parser.add_argument('--target', type=str, required=True, default='adspend', help='target feature in S or MS task')
parser.add_argument('--pred_len', type=int, required=True, default=7, help='prediction sequence length')

args = parser.parse_args()

df = pd.read_csv(os.path.join(args.root_path, args.data_path))
last_value = df[args.target].iloc[-1]
replicated_values = np.array([last_value for _ in range(args.pred_len)])
output_path = os.path.join('results', f'{args.target}.npy')
np.save(output_path, replicated_values)








