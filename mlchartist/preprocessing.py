# -*- coding: UTF-8 -*-


"""
Preprocessing Function Librairy
"""



import pandas as pd



def to_date(df, date_column):
    """
    Convert int of dated_col to to datetime object
    """
    dated_col = pd.to_datetime(df[date_column], format=(%Y%m%d))                #Use directly the right column as arg?
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


