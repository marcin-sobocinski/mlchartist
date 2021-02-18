import pandas as pd
import glob

from gcp import load_file_from_gcp, array_of_files_in_folder


def check_nrows(nrows=None, df=None):
    if nrows is None:
        return len(df)
    return nrows


def load_processed_data(nrows=10000, local=False, ticker_list=None, min_length=500, nasdaq100=True, gcp_credentials_path=None):
    joined_df = pd.DataFrame()
    if nasdaq100 == True:
        ticker_list = pd.read_csv('../raw_data/nasdaq100.csv', header=None)
        ticker_list = list(ticker_list.values.flatten())

    ## load from local machine
    if local == True:
        if ticker_list is not None:
            for ticker in ticker_list:
                df = pd.read_csv('../raw_data/processed/' + ticker.strip().lower() '.csv')
                if len(df) > min_length:
                    joined_df.append(df)
        else:
            csv_files = glob.glob('../raw_data/processed/*.csv')
            for ticker in csv_files:
                df = pd.read_csv(ticker)
                if len(df) > min_length:
                    joined_df.append(df)

    ## load from GCP
    elif local == False:
        if ticker_list is not None:
            for ticker in ticker_list:
                file_path = 'nasdaq_100_processed/' + ticker.strip().lower() '.csv'
                df = load_file_from_gcp(bucket_name='mlchartist-project', source_blob_name=file_path, credentials_path=gcp_credentials_path)
                if len(df) > min_length:
                    joined_df.append(df)
        else:
            csv_files = array_of_files_in_folder(bucket_name='mlchartist-project', 
                       prefix='nasdaq_100_processed/', credentials_path=gcp_credentials_path)
            for ticjer in csv_files:
                file_path = 'nasdaq_100_processed/' + ticker.strip().lower() '.csv'
                df = load_file_from_gcp(bucket_name='mlchartist-project', source_blob_name=file_path, credentials_path=gcp_credentials_path)
                if len(df) > min_length:
                    joined_df.append(df)
    
    return joined_df
            


