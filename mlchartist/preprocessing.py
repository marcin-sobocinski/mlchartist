# -*- coding: UTF-8 -*-


"""
Preprocessing Function Librairy
"""


import numpy as np
import pandas as pd



def to_date(df, date_column):
    """
    Convert int of dated_col to to datetime object
    """
    dated_col = pd.to_datetime(df[date_column], format=('%Y%m%d'))                #Use directly the right column as arg?
    return dated_col


def proper_name(col):
    """
    Mapping function for proper_col()
    """
    col = col.replace('>', '')
    col = col.replace('<', '')
    col = col.lower()
    return col


def proper_col(df):
    """
    Clean column '<TITLE>' to 'title'
    Remove '.US' from ticker col
    """
    proper_df = pd.DataFrame()

    for col in list(df):
        proper_col = proper_name(col)
        proper_df[proper_col] = df[col]

    proper_df = proper_df["ticker"].apply(lambda x: x.replace('.US', ''))
    return proper_df


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
