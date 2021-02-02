# -*- coding: UTF-8 -*-
"""
Generates input and output arrays out of single dataframe with input and target features
"""

import pandas as pd
import numpy as np

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



