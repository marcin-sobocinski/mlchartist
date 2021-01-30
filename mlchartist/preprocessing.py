# -*- coding: UTF-8 -*-


"""
Preprocessing Function Librairy
"""


import numpy as np
import pandas as pd
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator, MACD

# Column names cleaning
def to_date(df, date_column):
    """
    Convert int of dated_col to to datetime object, return converted column
    """
    dated_col = pd.to_datetime(df[date_column], format=('%Y%m%d'))                #Use directly the right column as arg?
    return dated_col


def proper_name(col):
    """
    Transform column name into lowercase with no '<>', return column name
    """
    col = col.replace('>', '')
    col = col.replace('<', '')
    col = col.lower()
    return col

# Ticker name cleaning
def proper_col(df):
    """
    Remove '.US' from ticker col, returns cleaned df
    """
    clean_df = df.copy()
    clean_df['ticker'] = clean_df['ticker'].str.rstrip('.US')
    return clean_df

# Target variable calculations
def calculate_real_returns(df):
    """
       Calculate Real Returns for 5, 10 & 20 Trading Days
       INPUT: Assumes the headers have been changed. Example: '<CLOSE>' is 'close'
    """
    returns_df = df.copy()
    ## 5 Trading Day Real Returns
    returns_df['5TD_return'] = (df['close'].shift(-5)/df['close'])-1
    ## 10 Trading Day Real Returns
    returns_df['10TD_return'] = (df['close'].shift(-10)/df['close'])-1
    ## 20 Trading Day Real
    returns_df['20TD_return'] = (df['close'].shift(-20)/df['close'])-1
    return returns_df

def calculate_log_returns(df):
    """
       Calculate Log Returns for 5, 10 & 20 Trading Days
       INPUT: Assumes the headers have been changed. Example: '<CLOSE>' is 'close'
    """
    returns_df = df.copy()
    ## 5 Trading Day Log Returns
    returns_df['5TD__log_return'] = np.log(returns_df['close']) - np.log(returns_df['close'].shift(-5))
    ## 10 Trading Day Log Returns
    returns_df['10TD__log_return'] = np.log(returns_df['close']) - np.log(returns_df['close'].shift(-10))
    ## 20 Trading Day Log Returns
    returns_df['20TD__log_return'] = np.log(returns_df['close']) - np.log(returns_df['close'].shift(-20))
    return returns_df

# Technical Indicators calculations

# input feature names:
CLOSE = "close"
OPEN = "open"
HIGH = "high"
LOW = "low"
VOLUME = "vol"

def get_indicators(df):
    """
        Add set of technical indicators to the dataframe, return original data frame with new features
    """
    feature_df = df.copy()
    feature_df['RSI'] = RSIIndicator(close=df[CLOSE]).rsi()
    feature_df['Stochastic'] = StochasticOscillator(high=df[HIGH], low=df[LOW], close=df[CLOSE]).stoch()
    feature_df['Stochastic_signal'] = StochasticOscillator(high=df[HIGH], low=df[LOW], close=df[CLOSE]).stoch_signal()
    feature_df['ADI'] = AccDistIndexIndicator(high=df[HIGH], low=df[LOW], close=df[CLOSE], volume=df[VOLUME]).acc_dist_index()
    feature_df['OBV'] = OnBalanceVolumeIndicator(close=df[CLOSE], volume=df[VOLUME]).on_balance_volume()
    feature_df['ATR'] = AverageTrueRange(high=df[HIGH], low=df[LOW], close=df[CLOSE]).average_true_range()
    feature_df['ADX'] = ADXIndicator(high=df[HIGH], low=df[LOW], close=df[CLOSE]).adx()
    feature_df['ADX_pos'] = ADXIndicator(high=df[HIGH], low=df[LOW], close=df[CLOSE]).adx_pos()
    feature_df['ADX_neg'] = ADXIndicator(high=df[HIGH], low=df[LOW], close=df[CLOSE]).adx_neg()
    feature_df['MACD'] = MACD(close=df[CLOSE]).macd()
    feature_df['MACD_diff'] = MACD(close=df[CLOSE]).macd_diff()
    feature_df['MACD_signal'] = MACD(close=df[CLOSE]).macd_signal()
    return feature_df


def train_test_split(df, test_set_size):
    """
    Split the preprocessed stock data file into a train and test dataset
    INPUT: the dataframe to be split, and size of the test set in months or years ('3M' or '2Y')
    OUTPUT: returns a train_set and test_set dataframe, index is set to the date
    
    EXAMPLE: train_set, test_set = train_test_split(input_df, '3Y')  --> puts last 3 years in test_set
    """
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'], format=('%Y-%m-%d'))
    test_set = df.sort_values(by="date",ascending=True).set_index("date").last(test_set_size)
    train_set = df.drop(df.tail(len(test_set)).index).set_index("date")
    test_set.reset_index(inplace=True)
    train_set.reset_index(inplace=True)
    return train_set, test_set


def returns_classification(return_column, returns_threshold):
    """
    Classify the returns versus a defined threshold, and returning either a 1 or 0
    INPUT: the dataframes column, and return threshold
    OUTPUT: returns a column with 1/0 binary classification 
    
    EXAMPLE: train_set['5TD_return_B'] = returns_classification(train_set['5TD_return'], 0.0006)
    """
    return (return_column > returns_threshold).astype(np.int)




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

