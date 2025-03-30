import numpy as np
import pandas as pd
import json

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from syntheval import SynthEval

pd.options.mode.chained_assignment = None

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()


if __name__ == '__main__':

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path

    real_path = f'synthetic/{dataname}/real.csv'
    test_path = f'synthetic/{dataname}/test.csv'

    data_dir = f'data/{dataname}' 

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    column_names = info['column_names']
    target_col_idx = info['target_col_idx'][0]
    target_column = column_names[target_col_idx]

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)
    test_data = pd.read_csv(test_path)

    S = SynthEval(real_data, holdout_dataframe=test_data)
    eval_df = S.evaluate(syn_data, target_column, "mia")

    # Filter for rows with 'mia_recall' and 'mia_precision'
    filtered_rows = eval_df[eval_df['metric'].isin(['mia_recall', 'mia_precision'])]

    # Extract values into variables
    mia_recall_val = filtered_rows.loc[filtered_rows['metric'] == 'mia_recall', 'val'].values[0]
    mia_recall_err = filtered_rows.loc[filtered_rows['metric'] == 'mia_recall', 'err'].values[0]
    mia_precision_val = filtered_rows.loc[filtered_rows['metric'] == 'mia_precision', 'val'].values[0]
    mia_precision_err = filtered_rows.loc[filtered_rows['metric'] == 'mia_precision', 'err'].values[0]

    # Print extracted variables
    
    print("mia_precision_val:", mia_precision_val)
    print("mia_precision_err:", mia_precision_err)
    print("mia_recall_val:", mia_recall_val)
    print("mia_recall_err:", mia_recall_err)
    
    save_dir = f'eval/privacy/{dataname}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = f'eval/privacy/{dataname}/{model}'

    print('Saving scores to ', f'{save_dir}.txt')
    with open(f'{save_dir}' + '.txt', 'w') as f:
        f.write(f'mia_precision_val={np.round(mia_precision_val, 4)}\n')
        f.write(f'mia_precision_err={np.round(mia_precision_err, 4)}\n')
        f.write(f'mia_recall_val={np.round(mia_recall_val, 4)}\n')
        f.write(f'mia_recall_err={np.round(mia_recall_err, 4)}\n')
        