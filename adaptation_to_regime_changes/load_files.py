import os

import scipy
import pandas as pd


# To load the files that have only two options (right, left) just call load_binary_choice_experiment_file(<PATH>),
# where PATH is the path to the folder with the data.
def load_binary_choice_experiment_files(path):
    files = [file for file in os.listdir(path) if ('_WS' in file) & ('training' not in file)]
    files = [file for file in files if (int(file[9:12].replace('_', '')) <= 53)]
    return load_files(path, files)


def generate_subject_df(data, subject):
    df = pd.DataFrame()
    df['state'] = data['S'].reshape(len(data['S']))
    df['subject'] = subject
    df['action'] = data['A'][0]
    df['correct'] = data['C'][0]
    df['reaction_time'] = data['RT'][0]
    df['reward'] = list(data['R'])
    return df


def load_files(path, files):
    df_list = []
    for file in files:
        data = scipy.io.loadmat(f'{path}/{file}')
        subject_df = generate_subject_df(data, int(file[9:12]))
        df_list.append(subject_df)
    return pd.concat(df_list)


