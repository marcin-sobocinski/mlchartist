import pandas as pd
import glob

from mlchartist.gcp import load_file_from_gcp, array_of_files_in_folder


def check_nrows(nrows=None, df=None):
    if nrows is None:
        return len(df)
    return nrows


def load_processed_data(nrows=10000, local=False, ticker_list=None, min_length=500, nasdaq100=False, gcp_credentials_path=None,
            nasdaq100_list_filepath='raw_data/nasdaq100.csv', processed_csvs_filepath='raw_data/processed/'):
    joined_df = pd.DataFrame()
    if ticker_list is None and nasdaq100 == True:
        if local == True:  ticker_list = pd.read_csv(nasdaq100_list_filepath, header=None)
        if local == False: ticker_list = load_file_from_gcp(bucket_name='mlchartist-project', source_blob_name='nasdaq100.csv', credentials_path=gcp_credentials_path)
        ticker_list = list(ticker_list.values.flatten())

    ## load from local machine
    if local == True:
        print('Loading Data From Local')
        if ticker_list is not None:
            for ticker in ticker_list:
                file_path = processed_csvs_filepath + ticker.strip().lower() + '.csv'
                df = pd.read_csv(file_path, nrows=nrows)
                if len(df) > min_length:
                    joined_df = joined_df.append(df)
        else:
            csv_files = glob.glob(processed_csvs_filepath + '*.csv')
            for ticker in csv_files:
                df = pd.read_csv(ticker, nrows=nrows)
                if len(df) > min_length:
                    joined_df = joined_df.append(df)

    ## load from GCP
    elif local == False:
        print('Loading Data From GCP')
        if ticker_list is not None:
            for ticker in ticker_list:
                file_path = 'nasdaq_100_processed/' + ticker.strip().lower() + '.csv'
                df = load_file_from_gcp(bucket_name='mlchartist-project', source_blob_name=file_path, credentials_path=gcp_credentials_path)
                if len(df) > min_length:
                    nrows = check_nrows(nrows=nrows, df=df)
                    reduced_df = df.iloc[0: nrows].copy()
                    joined_df = joined_df.append(reduced_df)
        else:
            csv_files = array_of_files_in_folder(bucket_name='mlchartist-project', 
                       prefix='nasdaq_100_processed/', credentials_path=gcp_credentials_path)
            for ticjer in csv_files:
                file_path = 'nasdaq_100_processed/' + ticker.strip().lower() + '.csv'
                df = load_file_from_gcp(bucket_name='mlchartist-project', source_blob_name=file_path, credentials_path=gcp_credentials_path)
                if len(df) > min_length:
                    nrows = check_nrows(nrows=nrows, df=df)
                    reduced_df = df.iloc[0: nrows].copy()
                    joined_df = joined_df.append(reduced_df)
    
    return joined_df
            


