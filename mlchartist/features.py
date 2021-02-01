# -*- coding: UTF-8 -*-


"""
Transforming raw data files into set of csv's with all features calculated
"""

import pandas as pd
import numpy as np
from mlchartist.preprocessing import to_date, proper_name, proper_col, calculate_real_returns, get_indicators
import os

def transform_file(filename):
    """
    Applies all preprocessing steps (preprocessing.py) to a single file,
    takes file path, returns dataframe
    """
    df = pd.read_csv(filename)
    df.columns = [proper_name(col) for col in df.columns]
    df['date'] = to_date(df, 'date')
    df = proper_col(df)
    df.drop(columns=['per', 'time', 'openint'], inplace=True)
    df = get_indicators(df)
    df_final = calculate_real_returns(df)
    df_final = calculate_past_returns(df_final)
    df_final = df_final.dropna().drop(columns = ['open', 'high','low','close', 'vol']).reset_index(drop=True)
    return df_final

def save_ticker(df, pathname):
    """
    Saves final dataframe to the pathname destination, assumes pathname exists
    """
    df.to_csv(pathname, index=False)

def build_data(raw_data_folder=r'../raw_data/data/daily/us/nasdaq stocks/', destination_path=r'../raw_data/processed/', len_hist=60):
    """
    Transforms and stores at destination_path all .txt files in raw_data_folder.
    The function assumes destination_path is a folder that exists!


    len_hist is a min number of rows in a file
    """
    files_changed = 0
    for subdir, dirs, files in os.walk(raw_data_folder):
        for filename in files:
            filepath = subdir + os.sep + filename
            if not subdir.endswith('.ipynb_checkpoints'):
                if filename.endswith('txt'):
                    with open(filepath) as f:
                        rows_num = sum(1 for line in f)
                        if rows_num >= len_hist:
                            df = transform_file(filepath)
                            new_name = filename.rstrip('.us.txt') + '.csv'
                            targetpath = destination_path + os.sep + new_name
                            save_ticker(df, targetpath)
                            files_changed += 1
    print(f'Number of files transformed {files_changed}')


if __name__ == '__main__':
    build_data()
