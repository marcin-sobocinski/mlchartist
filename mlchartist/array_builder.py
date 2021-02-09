# -*- coding: UTF-8 -*-
"""
Generates input and output arrays out of single dataframe with input and target features
"""

import pandas as pd
import numpy as np
import random
from mlchartist.preprocessing import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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


def build_randomised_arrays(df, time_window=5, stride=3, check_outliers=False, outlier_threshold=1, input_cols=['RSI', 'Stochastic', 'Stochastic_signal', 'ADI',
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
    outlier_count = 0
    for window_start in random_index:
        outlier = False
        df_slice = df_sorted.iloc[window_start: window_start + time_window]
        if check_outliers == True:
            for k, v in outlier_validation.items(): 
                if ((df_slice[k] < v[0]).any() == True) or ((df_slice[k] > v[1]).any() == True): outlier = True
        if df_slice.shape[0]==time_window and outlier==False:
            if outlier_count/max_num_windows >= outlier_threshold:
                return np.array([]), np.array([])
            input_array.append(np.array(df_slice[input_cols].values))
            target_array.append(np.array(df_slice[target_col].values))
        else: outlier_count+=1  
    return np.array(input_array), np.array(target_array)

from mlchartist.preprocessing import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from mlchartist.preprocessing import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import random
import pandas as pd
import numpy as np

def full_dataset_randomised_arrays_(df, 
                                         test_set_size='3Y', 
                                         time_window=5, 
                                         stride=3, 
                                         check_train_outliers=False, 
                                         check_test_outliers=False, 
                                         outlier_threshold=1, 
                                         input_cols=['RSI', 'Stochastic', 'Stochastic_signal', 'ADI','OBV', 'ATR', 'ADX', 
                                                     'ADX_pos', 'ADX_neg', 'MACD', 'MACD_diff', 'MACD_signal', '1D_past_return', 
                                                     '5D_past_return', '10D_past_return'], 
                                         target_col=['1D_past_return', '5D_past_return', '10D_past_return'], 
                                         outlier_validation={'ATR': [-100, 100], 'Stochastic': [0, 100], 
                                                             'Stochastic_signal': [-10, 110], '5D_past_return': [-0.5, 0.5]}):
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
       be included in the input array target_col (default = '5TD_return') - target variable, first 
       (newest) value for each input array
    target_col - all columns that should be included in target_col
        (default: target_col=['1D_past_return', '5D_past_return', '10D_past_return'])
    outlier_validation - a dict that sets the outlier checks to be completed. Enter data in the format:
        outlier_validation={'column_name': [lower_threshold, upper_threshold]} 
        Example: {'Stochastic': [0, 100], 'Stochastic_signal': [-10, 110], '5D_past_return': [-0.5, 0.5]}

    Return tuple (input_array, target_array).

    input_array dim: (number_of_samples x time_window x features_number)
    target_array dim: (number_of_samples x time_window x returns_numbder)
    """
    
    ## split into train/test split
    raw_train_set = pd.DataFrame()
    raw_test_set = pd.DataFrame()
    for ticker in df['ticker'].unique():
        company_df = df[df['ticker'] == ticker]
        temp_train_set, temp_test_set = train_test_split(company_df, test_set_size)
        raw_train_set = raw_train_set.append(temp_train_set)
        raw_test_set = raw_test_set.append(temp_test_set)
        
    ## create copy of train_set & fit scaler
    no_outlier_train_df = raw_train_set.copy()
    for k, v in outlier_validation.items(): 
        no_outlier_train_df = no_outlier_train_df[no_outlier_train_df[k].between(v[0], v[1])]
    scaler = RobustScaler()
    scaler.fit(no_outlier_train_df[input_cols])
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    stats2 = []
    stats = {}
    ## go company by company
    print(f"{df['ticker'].unique().size} Companies in Dataset")
    status_count = 0
    for ticker in df['ticker'].unique():
        status_count +=1
        stats[ticker] = {}
        print(f"Starting {ticker}: Company {status_count} of {df['ticker'].unique().size}")
        train_outlier_count = 0
        test_outlier_count = 0
        company_train_x_array = []
        company_train_y_array = []
        
        company_test_x_array = []
        company_test_y_array = []

        ## train
        company_train_df = raw_train_set[raw_train_set['ticker'] == ticker]
        company_train_sorted = company_train_df.sort_values('date', ascending=False)
        company_train_sorted.reset_index(drop=True, inplace=True)
        for row in range(0, len(company_train_sorted), stride):
            outlier = False
            df_slice = company_train_sorted.iloc[row: row + time_window].copy()
            ## check for outliers
            if check_train_outliers == True:
                for k, v in outlier_validation.items(): 
                    if ((df_slice[k] < v[0]).any() == True) or ((df_slice[k] > v[1]).any() == True): outlier = True
                
            if df_slice.shape[0]==time_window and outlier==False:
                ## scale the window
                df_slice.loc[:, input_cols] = scaler.transform(df_slice[input_cols])
                ## add to company array
                company_train_x_array.append(np.array(df_slice[input_cols].values))
                company_train_y_array.append(np.array(df_slice[target_col].values))
            else: train_outlier_count+=1
        
        if train_outlier_count/(len(company_train_sorted)/stride) <= outlier_threshold:
            stats[ticker]['train_possible_windows'] = (len(company_train_sorted)/stride)
            stats[ticker]['train_outliers'] = train_outlier_count
            stats[ticker]['train_windows'] = len(company_train_x_array)
            train_x.extend(company_train_x_array)
            train_y.extend(company_train_y_array)
            

        ## test
        company_test_df = raw_test_set[raw_test_set['ticker'] == ticker]
        company_test_sorted = company_test_df.sort_values('date', ascending=False)
        company_test_sorted.reset_index(drop=True, inplace=True)
        for row in range(0, len(company_test_sorted), stride):
            outlier = False
            df_slice = company_test_sorted.iloc[row: row + time_window].copy()
            ## check for outliers
            if check_test_outliers == True:
                for k, v in outlier_validation.items(): 
                    if ((df_slice[k] < v[0]).any() == True) or ((df_slice[k] > v[1]).any() == True): outlier = True
                
            if df_slice.shape[0]==time_window and outlier==False:
                ## scale the window
                df_slice.loc[:, input_cols] = scaler.transform(df_slice[input_cols])
                ## add to company array
                company_test_x_array.append(np.array(df_slice[input_cols].values))
                company_test_y_array.append(np.array(df_slice[target_col].values))
            else: test_outlier_count+=1
        
        if train_outlier_count/(len(company_train_sorted)/stride) <= outlier_threshold:
            stats[ticker]['test_possible_windows'] = (len(company_test_sorted)/stride)
            stats[ticker]['test_outliers'] = test_outlier_count
            stats[ticker]['test_windows'] = len(company_test_x_array)
            test_x.extend(company_test_x_array)
            test_y.extend(company_test_y_array)
    
    print('All Companies Completed')
    print('')
    print('Processing Stats:', stats)
    
    ## shuffle arrays
    output_train_x = []
    output_train_y = []
    index_list = random.sample(range(len(train_x)), len(train_x))
    for i in index_list:
        output_train_x.append(train_x[i])
        output_train_y.append(train_y[i])
    return np.array(output_train_x), np.array(output_train_y), np.array(test_x), np.array(test_y), scaler

