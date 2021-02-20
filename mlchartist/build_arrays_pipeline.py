from mlchartist.array_builder import full_dataset_randomised_arrays
from mlchartist.data import load_processed_data
from mlchartist.gcp import load_gcp_credentials
from google.cloud import storage



class BuildArrays(object):

    def __init__(self, unsplit_df=None, **kwargs):
        self.input_df = unsplit_df
        self.kwargs = kwargs
        ## outputs
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.scaler = None

    def build_arrays(self):
        """return estimator"""
        self.train_x, self.train_y, self.test_x, self.test_y, self.scaler = full_dataset_randomised_arrays(
                                                                                    unsplit_df=self.input_df, **self.kwargs)

    def save(self, local=True, scaler=True, numpy_arrays=True):
        """define pipeline here, define it as a class attribute"""
        pass

    def output(self):
        """evaluates the pipeline on df_test"""
        return self.train_x, self.train_y, self.test_x, self.test_y, self.scaler


if __name__ == "__main__":
    ##warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    experiment = "mlchartist_set_YOURNAME"
    gcp_details = load_gcp_credentials()
    data_params = dict(nrows=10000, 
                        local=False, 
                        ticker_list=['AAPL', 'TSLA'], 
                        min_length=500, 
                        nasdaq100=False, 
                        gcp_credentials_path=gcp_details)
    print("############   Loading Data   ############")
    df = load_processed_data(**data_params)

    # Build Numpy Arrays --> Parameters
    INPUT_COLS = ['RSI', 'Stochastic', 'Stochastic_signal', 'ADI','OBV', 'ATR', 'ADX', 'ADX_pos', 'ADX_neg', 'MACD', 'MACD_diff',
              'MACD_signal', '5TD_return', '10TD_return', '20TD_return']
    TARGET_COLS=['5TD_return', '10TD_return', '20TD_return']
    outlier_validation={'5TD_return': [-0.5, 0.5]}
    stride = 100
    build_array_params = dict(  stride=stride,
                                input_cols=INPUT_COLS, 
                                outlier_threshold=1, 
                                outlier_validation=outlier_validation, 
                                check_train_outliers=True,
                                check_test_outliers=False, 
                                target_col=TARGET_COLS, 
                                time_window=6,
                                test_set_size=500
        
    )
    arrays_pipeline = BuildArrays(unsplit_df=df, **build_array_params)
    del df
    print("############  Building Arrays model   ############")
    arrays_pipeline.build_arrays()
    print("############  Evaluating model ############")
    ##arrays_pipeline.save()
    print("############   Saving model    ############")
    train_x, train_y, test_x, test_y, scaler = arrays_pipeline.output()
    print('')
    print('')
    print('### Stats ###')
    print('train_x', train_x.shape)
    print('train_y', train_y.shape)
    print('test_x', test_x.shape)
    print('test_y', test_y.shape)
    print('scaler', scaler)



