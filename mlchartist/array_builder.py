# -*- coding: UTF-8 -*-
"""
Generates input and output arrays out of single dataframe with input and target features
"""

import pandas as pd
import numpy as np
import random

def build_arrays(df, time_window=5, stride=3, input_cols=['RSI', 'Stochastic', 'Stochastic_signal', 'ADI',
       'OBV', 'ATR', 'ADX', 'ADX_pos', 'ADX_neg', 'MACD', 'MACD_diff',
       'MACD_signal', '1D_past_return', '5D_past_return', '10D_past_return'] , target_col='5TD_return'):
    """
    A function to transform dataframe into input and output arrays.

    Takes:
    df - input dataframe
    time_window (default=5) - time series length
    stride (default=3) - a step for moving window across dataframe rows
    input_cols (default = 'RSI', 'Stochastic', 'Stochastic_signal', 'ADI',
       'OBV', 'ATR', 'ADX', 'ADX_pos', 'ADX_neg', 'MACD', 'MACD_diff',
       'MACD_signal', '1D_past_return', '5D_past_return', '10D_past_return']) - all input features, that should be included in the input array
    target_col (default = '5TD_return') - target variable, first (newest) value for each input array


    Return tuple (input_array, target_array).

    input_array dim: (number_of_samples x time_window x features_number)
    target_array dim: number_of_samples
    """

    input_array = []
    target_array = []
    df_sorted = df.sort_values('date', ascending=False)
    df_sorted.reset_index(drop=True, inplace=True)
    for row in range(0, len(df), stride):
        df_slice = df_sorted.iloc[row: row + time_window]
        if df_slice.shape[0]==time_window:
            input_array.append(np.array(df_slice[input_cols].values))
            target_array.append(df_slice[target_col].iloc[0])
    return np.array(input_array), np.array(target_array)


## Ian's window function
def window_dataframe(df, window=30, stride_size=5, target=['5TD_return'], feature_cols=['RSI', 'Stochastic', 'Stochastic_signal', 
        'ADI', 'OBV', 'ATR', 'ADX', 'ADX_pos', 'ADX_neg', 'MACD', 'MACD_diff', 'MACD_signal']):
    """
    Turns the input dataframe into an array of windowed arrays
    INPUT: the input dataframe, window size, stride size, target column, feature columns
    OUTPUT: array of windowed arrays 
    
    EXAMPLE: windowed_array = window_dataframe(train_set)
    """
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'], format=('%Y-%m-%d'))
    inverse_df = df.sort_values(by="date", ascending=False)
    feature_array = []
    target_array = []
    for column in inverse_df:
        if column in feature_cols: 
            feature_array.append(window_column(inverse_df[column], window, stride_size))
            
        elif column in target:
            target_array.append(window_column(inverse_df[column], window, stride_size))
            
    
    return np.array(feature_array), np.array(target_array)


## this function is called in window_dataframe function
def window_column(df_series, window_size=30, stride_size=5):
    """
    Turns data series into array of windowed arrays
    INPUT: the input data series, window size, stride size
    OUTPUT: array of windowed arrays 
    
    EXAMPLE: y = window_column(train_set['RSI'], 30, 5)
    """
    np_array = df_series.to_numpy()
    nrows = ((np_array.size-window_size)//stride_size)+1
    n = np_array.strides[0]
    return np.lib.stride_tricks.as_strided(
        np_array, shape=(nrows, window_size), strides=(stride_size*n, n))


def build_randomised_arrays(df, time_window=5, stride=3, check_outliers=False, input_cols=['RSI', 'Stochastic', 'Stochastic_signal', 'ADI',
       'OBV', 'ATR', 'ADX', 'ADX_pos', 'ADX_neg', 'MACD', 'MACD_diff',
       'MACD_signal', '1D_past_return', '5D_past_return', '10D_past_return'], target_col=['1D_past_return', '5D_past_return', '10D_past_return'], 
        outlier_validation={'ATR': [-100, 100], 'Stochastic': [0, 100], 'Stochastic_signal': [-10, 110], '5D_past_return': [-0.5, 0.5]}):
    """
    A function to transform dataframe into input and output arrays.

    Takes:
    df - input dataframe
    time_window (default=5) - time series length
    stride (default=3) - controls the number of windows taken (i.e. max_num_windows = len(df)/strides)
    check_outliers (default=False) - controls whether it checks each window for outliers or not
    input_cols (default = 'RSI', 'Stochastic', 'Stochastic_signal', 'ADI',
       'OBV', 'ATR', 'ADX', 'ADX_pos', 'ADX_neg', 'MACD', 'MACD_diff',
       'MACD_signal', '1D_past_return', '5D_past_return', '10D_past_return']) - all input features, that should 
       be included in the input array target_col (default = '5TD_return') - target variable, first (newest) value for each input array
    target_col - all columns that should be included in target_col
        (default: target_col=['1D_past_return', '5D_past_return', '10D_past_return'])
    outlier_validation - a dict that sets the outlier checks to be completed. Enter data in the format:
        outlier_validation={'column_name': [lower_threshold, upper_threshold]} 
        Example: {'Stochastic': [0, 100], 'Stochastic_signal': [-10, 110], '5D_past_return': [-0.5, 0.5]}

    Return tuple (input_array, target_array).

    input_array dim: (number_of_samples x time_window x features_number)
    target_array dim: (number_of_samples x time_window x returns_numbder)
    """

    input_array = []
    target_array = []
    df_sorted = df.sort_values('date', ascending=False)
    df_sorted.reset_index(drop=True, inplace=True)    
    max_num_windows = len(df)/stride
    random_index = []
    for i in range(int(max_num_windows)):
        r=random.randint(time_window, len(df)- time_window)
        if r not in random_index: random_index.append(r)
    
    for window_start in random_index:
        outlier = False
        df_slice = df_sorted.iloc[window_start: window_start + time_window]
        if check_outliers == True:
            for k, v in outlier_validation.items(): 
                if ((df_slice[k] < v[0]).any() == True) or ((df_slice[k] > v[1]).any() == True): outlier = True
        if df_slice.shape[0]==time_window and outlier==False:
            input_array.append(np.array(df_slice[input_cols].values))
            target_array.append(np.array(df_slice[target_col].values))
    
    return np.array(input_array), np.array(target_array)

