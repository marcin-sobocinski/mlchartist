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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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
    clean_df['ticker'] = clean_df['ticker'].str[:-3]
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

def calculate_past_returns(df):
    """
       Calculate Past Returns for 1, 5 & 10 Trading Days
       INPUT: Assumes the headers have been changed. Example: '<CLOSE>' is 'close'
    """
    returns_df = df.copy()
    ## 5 Trading Day Past Returns
    returns_df['1D_past_return'] = (df['close']/df['close'].shift(1))-1
    ## 10 Trading Day Past Returns
    returns_df['5D_past_return'] = (df['close']/df['close'].shift(5))-1
    ## 20 Trading Day Past Returns
    returns_df['10D_past_return'] = (df['close']/df['close'].shift(10))-1
    return returns_df

def train_test_split(input_df, test_set_size):
    """
    Split the preprocessed stock data file into a train and test dataset
    INPUT: the dataframe to be split, and size of the test set in the number of rows
    OUTPUT: returns a train_set and test_set dataframe, index is set to the date

    EXAMPLE: train_set, test_set = train_test_split(input_df, 500) 
    """
    df = input_df.copy()
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'], format=('%Y-%m-%d'))
    df = df.sort_values(by="date",ascending=False)
    test_set = df.iloc[0: test_set_size].copy()
    train_set = df.iloc[test_set_size: ].copy()
    return train_set, test_set

def returns_classification(return_column, returns_threshold):
    """
    Classify the returns versus a defined threshold, and returning either a 1 or 0
    INPUT: the dataframes column, and return threshold
    OUTPUT: returns a column with 1/0 binary classification

    EXAMPLE: train_set['5TD_return_B'] = returns_classification(train_set['5TD_return'], 0.0006)
    """
    return (return_column > returns_threshold).astype(np.int)


def std_scaler(df):
    """
    Scale the data with SKlearn StandardScaler
    """
    scaler = StandardScaler()
    col_to_scale_df = df.drop(columns=['ticker', 'date', '5TD_return', '10TD_return', '20TD_return'])
    col_to_scale = list(col_to_scale_df)
    scaled_df = df
    for col in col_to_scale:
        scaled_df[col] = scaler.fit_transform(scaled_df[[col]])
    return scaled_df, scaler


def thresholds_encoding(df, r5d=0.0006, same_thresholds=True, r10d=0.0012, r20d=0.0024):
    """
    Binary encode the 5, 10 and 20 days return columns according to the thresholds

    INPUT: dataframe with '5TD_return', '10TD_return' and '20TD_return' columns
    OUTPUT: dataframe with binary encoded aforementionned columns
            '5D_return_bin', '10D_return_bin' and '20D_return_bin'

    If the thresolds returns are the same on a yearly basis for the different period use:
                r10d = r5d * 2
            and
                r20d = r10d * 2
            keep same_thresholds=True
        Otherwise, define manually r10d and r20d
    """
    wk_df = df.copy()

    if same_thresholds:
        r10d = r5d * 2
        r20d = r10d * 2

    wk_df['5D_return_bin'] = wk_df['5TD_return'].apply(lambda x: 1 if x > r5d else 0)
    wk_df['10D_return_bin'] = wk_df['10TD_return'].apply(lambda x: 1 if x > r10d else 0)
    wk_df['20D_return_bin'] = wk_df['20TD_return'].apply(lambda x: 1 if x > r20d else 0)

    return wk_df

def fit_train_scaler(train_df,
                    outlier_validation={'5TD_return': [-0.5, 0.5]},
                    input_cols=['RSI', 'Stochastic', 'Stochastic_signal', 'ADI','OBV', 'ATR', 'ADX',
                                    'ADX_pos', 'ADX_neg', 'MACD', 'MACD_diff', 'MACD_signal', '5TD_return',
                                    '10TD_return', '20TD_return']):
    """
    Fits Robust Scaler on train set and returns the scaler
    INPUT: the dataframe to used to fit scaler, the outlier_validation thresholds, the columns for the scaler to be
            applied too.
    OUTPUT: fitted scaler
    """
    no_outlier_train_df = train_df.copy()
    for k, v in outlier_validation.items():
        no_outlier_train_df = no_outlier_train_df[no_outlier_train_df[k].between(v[0], v[1])]
    scaler = RobustScaler()
    scaler.fit(no_outlier_train_df[input_cols])
    return scaler

def train_test_split_multiple_companies(df, test_set_size):
    """
    Split the preprocessed stock data of multiple companies into a train and test dataset
    INPUT: the dataframe to be split, and size of the test set in number of rows
    OUTPUT: returns a train_set and test_set dataframe, index is set to the date

    EXAMPLE: train_set, test_set = train_test_split_multiple_companies(input_df, 500) 
    """
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    for ticker in df['ticker'].unique():
        company_df = df[df['ticker'] == ticker]
        temp_train_set, temp_test_set = train_test_split(company_df, test_set_size)
        train_set = train_set.append(temp_train_set)
        test_set = test_set.append(temp_test_set)
    return train_set, test_set
