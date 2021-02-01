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

dfdef calculate_past_returns(df):
    """
       Calculate Past Returns for 1, 5 & 10 Trading Days
       INPUT: Assumes the headers have been changed. Example: '<CLOSE>' is 'close'
    """
    returns_df = df.copy()
    ## 5 Trading Day Real Returns
    returns_df['1D_past_return'] = (df['close']/df['close'].shift(1))-1
    ## 10 Trading Day Real Returns
    returns_df['5D_past_return'] = (df['close']/df['close'].shift(5))-1
    ## 20 Trading Day Real
    returns_df['10D_past_return'] = (df['close']/df['close'].shift(10))-1
    return returns_df










