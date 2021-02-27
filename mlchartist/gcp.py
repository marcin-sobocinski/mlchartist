from google.cloud import storage
from io import BytesIO
import pandas as pd
import configparser
import os

from mlchartist.params import BUCKET_NAME, MODEL_NAME, BUCKET_PREPROCESSED_FOLDER, MODEL_VERSION


def download_file_to_folder(bucket_name=BUCKET_NAME, source_blob_name=None, destination_file_name=None, credentials_path=None):
    if credentials_path is not None:
            storage_client = storage.Client.from_service_account_json(credentials_path)
    else:
        storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.get_blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def array_of_files_in_folder(bucket_name=BUCKET_NAME, prefix=None, delimiter=None, credentials_path=None):
    if credentials_path is not None:
            storage_client = storage.Client.from_service_account_json(credentials_path)
    else:
        storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    file_list = []
    for blob in blobs:
        file = blob.name.split(prefix)[1]
        if file != '':
            file_list.append(file)  
    return file_list


def load_file_from_gcp(bucket_name=BUCKET_NAME, source_blob_name=None, credentials_path=None):
    if credentials_path is not None:
            storage_client = storage.Client.from_service_account_json(credentials_path)
    else:
        storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.get_blob(source_blob_name)
    content = blob.download_as_string()
    df = pd.read_csv(BytesIO(content))
    return df

def load_gcp_credentials(config_path='config/credentials.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    gcp_path = config['gcp']['credentials_path']
    return gcp_path


def upload_model_gcp(model_version=MODEL_VERSION, bucket_name=BUCKET_NAME, rm=False, credentials_path=None):
    if credentials_path is not None:
            storage_client = storage.Client.from_service_account_json(credentials_path)
    else:
        storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    model = 'models/test_model.joblib'
    destination_blob_name = f'models/{MODEL_NAME}/versions/{model_version}/{model}.joblib'
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(model)
    print(f"Model {model} uploaded to {destination_blob_name}")
    if rm:
        os.remove('models/test_model.joblib')
