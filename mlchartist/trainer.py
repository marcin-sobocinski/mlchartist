from mlchartist.gcp import load_gcp_credentials
from mlchartist.data import load_processed_data

class Trainer(object):

    def __init__(self, unsplit_df=None, **kwargs):
        self.input_df = unsplit_df
        self.kwargs = kwargs
        ## outputs
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.scaler = None
        self.model = None


    def build_numpy_arrays(self):
        pass

    def load_numpy_arrays(self):
        pass

    def save_numpy_arrays(self):
        pass

    def define_model(self):
        """return model"""
        pass

    def fit_model(self):
        """define pipeline here, define it as a class attribute"""

    def save_model(self, X_test, y_test):
        """evaluates the pipeline on df_test"""

    def evaluation_model(self):
        pass

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
    print("############   Loading Processed CSVs   ############")
    df = load_processed_data(**data_params)
    print("############   Generating Numpy Arrays   ############")
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
    arrays_pipeline.build_arrays()
    del df
    print("shape: {}".format(X_train.shape))
    print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
    # Train and save model, locally and
    t = Trainer(X=X_train, y=y_train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model()
