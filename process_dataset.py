import numpy as np
import pandas as pd
import os
import sys
import json
import argparse

TYPE_TRANSFORM ={
    'float', np.float32,
    'str', str,
    'int', int
}

INFO_PATH = 'data/Info'

parser = argparse.ArgumentParser(description='process dataset')

# General configs
parser.add_argument('--dataname', type=str, default=None, help='Name of dataset.')
args = parser.parse_args()

def preprocess_beijing():
    with open(f'{INFO_PATH}/beijing.json', 'r') as f:
        info = json.load(f)
    
    data_path = info['raw_data_path']

    data_df = pd.read_csv(data_path)
    columns = data_df.columns

    data_df = data_df[columns[1:]]

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info['data_path'], index = False)

def preprocess_news():
    with open(f'{INFO_PATH}/news.json', 'r') as f:
        info = json.load(f)

    data_path = info['raw_data_path']
    data_df = pd.read_csv(data_path)
    data_df = data_df.drop('url', axis=1)

    columns = np.array(data_df.columns.tolist())

    cat_columns1 = columns[list(range(12,18))]
    cat_columns2 = columns[list(range(30,38))]

    cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis = 1)
    cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis = 1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)

    data_df['data_channel'] = cat_col1
    data_df['weekday'] = cat_col2
    
    data_save_path = 'data/news/news.csv'
    data_df.to_csv(f'{data_save_path}', index = False)

    columns = np.array(data_df.columns.tolist())
    num_columns = columns[list(range(45))]
    cat_columns = ['data_channel', 'weekday']
    target_columns = columns[[45]]

    info['num_col_idx'] = list(range(45))
    info['cat_col_idx'] = [46, 47]
    info['target_col_idx'] = [45]
    info['data_path'] = data_save_path
    
    name = 'news'
    with open(f'{INFO_PATH}/{name}.json', 'w') as file:
        json.dump(info, file, indent=4)

def preprocess_stroke():
    with open(f'{INFO_PATH}/stroke.json', 'r') as f:
        info = json.load(f)
    
    data_path = info['data_path']

    data_df = pd.read_csv(data_path)
    if 'id' in data_df.columns:
        data_df = data_df.drop('id', axis=1)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info['data_path'], index = False)
    
    
def preprocess_diabetes():
    with open(f'{INFO_PATH}/diabetes.json', 'r') as f:
        info = json.load(f)
    
    all_cols = pd.read_csv('data/diabetes/diabetic_data_og.csv', nrows=1).columns.tolist()
    used_cols = [col for col in all_cols if col != 'payer_code']

    data_df = pd.read_csv(
        'data/diabetes/diabetic_data_og.csv',
        usecols=used_cols,
        na_values=['?']
    )
    
    missing_percentage = data_df.isnull().mean() * 100
    sorted_missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
    print("Column-wise Missing Feature Percentage (Sorted):")
    print(sorted_missing_percentage)
    
    # Specify columns to drop based on missing percentages
    columns_to_drop = [
        'encounter_id',
        'admission_type_id',
        'discharge_disposition_id',
        'admission_source_id',
        'patient_nbr',
        'weight',
        'max_glu_serum',
        'A1Cresult',
        'medical_specialty',
        'metformin-pioglitazone',
        'metformin-rosiglitazone',
        'glimepiride-pioglitazone',
        'acetohexamide',
        'chlorpropamide',
        'citoglipton',
        'examide',
        'tolazamide',
        'diag_1',
        'diag_2',
        'diag_3',
    ]

    print(f"Columns to drop: {columns_to_drop}")
    
    data_df = data_df.drop(columns=columns_to_drop)
    data_df = data_df.dropna()
    data_df = data_df.loc[data_df['gender'] != 'Unknown/Invalid']
    
    data_df = data_df[data_df['glyburide-metformin'].isin(data_df['glyburide-metformin'].value_counts()[data_df['glyburide-metformin'].value_counts() >= 30].index)]
    data_df = data_df[data_df['miglitol'].isin(data_df['miglitol'].value_counts()[data_df['miglitol'].value_counts() >= 30].index)]
    data_df = data_df[data_df['acarbose'].isin(data_df['acarbose'].value_counts()[data_df['acarbose'].value_counts() >= 10].index)]
    
    for column in data_df.columns:
        print(f"Feature: {column}")
        print(data_df[column].value_counts())
        print("-" * 50)  # Separator for readability
        
    print(data_df.shape)
    
    data_df.to_csv(info['data_path'], index = False)
    
    
def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train = 0, num_test = 0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)


    seed = 1234

    while True:
        print(f"testing seed {seed}")
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                print(f'Column {i} has missing values in training set')
                exit()
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1
        
    return train_df, test_df, seed    


def process_data(name):

    if name == 'news':
        preprocess_news()
    elif name == 'beijing':
        preprocess_beijing()
    elif name == 'stroke':
        preprocess_stroke()
    elif name == 'diabetes':
        preprocess_diabetes()

    with open(f'{INFO_PATH}/{name}.json', 'r') as f:
        info = json.load(f)

    data_path = info['data_path']

    if name == 'bank' or name == 'cardio':
        data_df = pd.read_csv(data_path, sep=';')

    elif info['file_type'] == 'csv':
            data_df = pd.read_csv(data_path, header = info['header'])

    elif info['file_type'] == 'xls':
        data_df = pd.read_excel(data_path, sheet_name='Data', header=1)
        data_df = data_df.drop('ID', axis=1)

    num_data = data_df.shape[0]

    column_names = info['column_names'] if info['column_names'] else data_df.columns.tolist()
 
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]

    # max_cardinality = 0
    # max_cardinality_column = None
    # for col in cat_columns:
    #     cardinality = data_df[col].nunique()  # Number of unique values
    #     if cardinality > max_cardinality:
    #         max_cardinality = cardinality
    #         max_cardinality_column = col

    # print(f"Column with maximum cardinality: {max_cardinality_column} ({max_cardinality} unique values)")

    target_columns = [column_names[i] for i in target_col_idx]

    if info['test_path']:

        # if testing data is given
        test_path = info['test_path']

        with open(test_path, 'r') as f:
            lines = f.readlines()[1:]
            test_save_path = f'data/{name}/test.data'
            if not os.path.exists(test_save_path):
                with open(test_save_path, 'a') as f1:     
                    for line in lines:
                        save_line = line.strip('\n').strip('.')
                        f1.write(f'{save_line}\n')

        test_df = pd.read_csv(test_save_path, header = None)
        train_df = data_df
        
    elif name == 'diabetes':
        num_train = int(num_data*0.8)
        num_test = num_data - num_train

        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)

    else:  
        # Train / Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)

        num_train = int(num_data*0.9)
        num_test = num_data - num_train

        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)


    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))
    print(num_data)
    print("name, data_df.shape, train_df.shape, test_df.shape, data_df.shape")
    print(name, data_df.shape, train_df.shape, test_df.shape, data_df.shape)

    col_info = {}
    
    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'numerical'
        col_info['max'] = float(train_df[col_idx].max())
        col_info['min'] = float(train_df[col_idx].min())
     
    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'
        col_info['categorizes'] = list(set(train_df[col_idx]))    

    for col_idx in target_col_idx:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info['type'] = 'numerical'
            col_info['max'] = float(train_df[col_idx].max())
            col_info['min'] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info['type'] = 'categorical'
            col_info['categorizes'] = list(set(train_df[col_idx]))      

    info['column_info'] = col_info

    train_df.rename(columns = idx_name_mapping, inplace=True)
    test_df.rename(columns = idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == '?', col] = 'nan'
    for col in num_columns:
        test_df.loc[test_df[col] == '?', col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == '?', col] = 'nan'


    
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()

    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

 
    save_dir = f'data/{name}'
    np.save(f'{save_dir}/X_num_train.npy', X_num_train)
    np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
    np.save(f'{save_dir}/y_train.npy', y_train)

    np.save(f'{save_dir}/X_num_test.npy', X_num_test)
    np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
    np.save(f'{save_dir}/y_test.npy', y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)

    print('Saving Data...')

    train_df.to_csv(f'{save_dir}/train.csv', index = False)
    test_df.to_csv(f'{save_dir}/test.csv', index = False)

    if not os.path.exists(f'synthetic/{name}'):
        os.makedirs(f'synthetic/{name}')
    
    train_df.to_csv(f'synthetic/{name}/real.csv', index = False)
    test_df.to_csv(f'synthetic/{name}/test.csv', index = False)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)

    info['column_names'] = column_names
    info['train_num'] = train_df.shape[0]
    info['test_num'] = test_df.shape[0]

    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping 

    metadata = {'columns': {}}
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    for i in num_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'numerical'
        metadata['columns'][i]['computer_representation'] = 'Float'

    for i in cat_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'categorical'


    if task_type == 'regression':
        
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'numerical'
            metadata['columns'][i]['computer_representation'] = 'Float'

    else:
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'categorical'

    info['metadata'] = metadata

    with open(f'{save_dir}/info.json', 'w') as file:
        json.dump(info, file, indent=4)

    print(f'Processing and Saving {name} Successfully!')

    print("Dataset:", name)
    # print('Total', info['train_num'] + info['test_num'])
    # print('Train', int((info['train_num'])))
    # print('Test/Validation', int((info['test_num'])*0.5))
    print('Total', info['train_num'] + info['test_num'])
    print('Train', info['train_num'] - info['test_num'])
    print('Test', info['test_num'])
    print('Validation', info['test_num'])
    # print('Validation', int((info['train_num'] + info['test_num'])*0.1))
    if info['task_type'] == 'regression':
        num = len(info['num_col_idx'] + info['target_col_idx'])
        cat = len(info['cat_col_idx'])
    else:
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
    print('Num', num)
    print('Cat', cat)

# def process_census_synthetic(name):

#     with open(f'{INFO_PATH}/{name}.json', 'r') as f:
#         info = json.load(f)

#     data_path = info['data_path']

#     if info['file_type'] == 'csv':
#             data_df = pd.read_csv(data_path, header = info['header'])

#     num_data = data_df.shape[0]

#     column_names = info['column_names'] if info['column_names'] else data_df.columns.tolist()
 
#     num_col_idx = info['num_col_idx']
#     cat_col_idx = info['cat_col_idx']
#     target_col_idx = info['target_col_idx']

#     idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

#     num_columns = [column_names[i] for i in num_col_idx]
#     cat_columns = [column_names[i] for i in cat_col_idx]
#     target_columns = [column_names[i] for i in target_col_idx]

#     train_df = pd.read_csv('synthetic/census_synthetic/real.csv')
#     test_df = pd.read_csv('synthetic/census_synthetic/test.csv')

#     train_df.columns = range(len(train_df.columns))
#     test_df.columns = range(len(test_df.columns))
#     print(num_data)
#     print("name, data_df.shape, train_df.shape, test_df.shape, data_df.shape")
#     print(name, data_df.shape, train_df.shape, test_df.shape, data_df.shape)

#     col_info = {}
    
#     for col_idx in num_col_idx:
#         col_info[col_idx] = {}
#         col_info['type'] = 'numerical'
#         col_info['max'] = float(train_df[col_idx].max())
#         col_info['min'] = float(train_df[col_idx].min())
     
#     for col_idx in cat_col_idx:
#         col_info[col_idx] = {}
#         col_info['type'] = 'categorical'
#         col_info['categorizes'] = list(set(train_df[col_idx]))    

#     for col_idx in target_col_idx:
#         if info['task_type'] == 'regression':
#             col_info[col_idx] = {}
#             col_info['type'] = 'numerical'
#             col_info['max'] = float(train_df[col_idx].max())
#             col_info['min'] = float(train_df[col_idx].min())
#         else:
#             col_info[col_idx] = {}
#             col_info['type'] = 'categorical'
#             col_info['categorizes'] = list(set(train_df[col_idx]))      

#     info['column_info'] = col_info

#     train_df.rename(columns = idx_name_mapping, inplace=True)
#     test_df.rename(columns = idx_name_mapping, inplace=True)

#     for col in num_columns:
#         train_df.loc[train_df[col] == '?', col] = np.nan
#     for col in cat_columns:
#         train_df.loc[train_df[col] == '?', col] = 'nan'
#     for col in num_columns:
#         test_df.loc[test_df[col] == '?', col] = np.nan
#     for col in cat_columns:
#         test_df.loc[test_df[col] == '?', col] = 'nan'


    
#     X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
#     X_cat_train = train_df[cat_columns].to_numpy()
#     y_train = train_df[target_columns].to_numpy()

#     X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
#     X_cat_test = test_df[cat_columns].to_numpy()
#     y_test = test_df[target_columns].to_numpy()

 
#     save_dir = f'data/{name}'
#     np.save(f'{save_dir}/X_num_train.npy', X_num_train)
#     np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
#     np.save(f'{save_dir}/y_train.npy', y_train)

#     np.save(f'{save_dir}/X_num_test.npy', X_num_test)
#     np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
#     np.save(f'{save_dir}/y_test.npy', y_test)

#     train_df[num_columns] = train_df[num_columns].astype(np.float32)
#     test_df[num_columns] = test_df[num_columns].astype(np.float32)

#     print('Saving Data...')

#     train_df.to_csv(f'{save_dir}/train.csv', index = False)
#     test_df.to_csv(f'{save_dir}/test.csv', index = False)

#     if not os.path.exists(f'synthetic/{name}'):
#         os.makedirs(f'synthetic/{name}')
    
#     train_df.to_csv(f'synthetic/{name}/real.csv', index = False)
#     test_df.to_csv(f'synthetic/{name}/test.csv', index = False)

#     print('Numerical', X_num_train.shape)
#     print('Categorical', X_cat_train.shape)

#     info['column_names'] = column_names
#     info['train_num'] = train_df.shape[0]
#     info['test_num'] = test_df.shape[0]

#     info['idx_mapping'] = idx_mapping
#     info['inverse_idx_mapping'] = inverse_idx_mapping
#     info['idx_name_mapping'] = idx_name_mapping 

#     metadata = {'columns': {}}
#     task_type = info['task_type']
#     num_col_idx = info['num_col_idx']
#     cat_col_idx = info['cat_col_idx']
#     target_col_idx = info['target_col_idx']

#     for i in num_col_idx:
#         metadata['columns'][i] = {}
#         metadata['columns'][i]['sdtype'] = 'numerical'
#         metadata['columns'][i]['computer_representation'] = 'Float'

#     for i in cat_col_idx:
#         metadata['columns'][i] = {}
#         metadata['columns'][i]['sdtype'] = 'categorical'


#     if task_type == 'regression':
        
#         for i in target_col_idx:
#             metadata['columns'][i] = {}
#             metadata['columns'][i]['sdtype'] = 'numerical'
#             metadata['columns'][i]['computer_representation'] = 'Float'

#     else:
#         for i in target_col_idx:
#             metadata['columns'][i] = {}
#             metadata['columns'][i]['sdtype'] = 'categorical'

#     info['metadata'] = metadata

#     with open(f'{save_dir}/info.json', 'w') as file:
#         json.dump(info, file, indent=4)

#     print(f'Processing and Saving {name} Successfully!')

#     print("Dataset:", name)
#     print('Total', info['train_num'] + info['test_num'])
#     print('Train/Validation', info['train_num'])
#     print('Test', info['test_num'])
#     # print('Validation', int((info['train_num'] + info['test_num'])*0.1))
#     if info['task_type'] == 'regression':
#         num = len(info['num_col_idx'] + info['target_col_idx'])
#         cat = len(info['cat_col_idx'])
#     else:
#         cat = len(info['cat_col_idx'] + info['target_col_idx'])
#         num = len(info['num_col_idx'])
#     print('Num', num)
#     print('Cat', cat)


if __name__ == "__main__":
    # if args.dataname == 'census_synthetic':
        # process_census_synthetic(args.dataname)
    if args.dataname:
        process_data(args.dataname)
    else:
        # for name in ['adult', 'default', 'shoppers', 'beijing', 'news', 'stroke', 'diabetes']:
        for name in ['shoppers']:
            process_data(name)
