from mlchartist.gcp import load_gcp_credentials, upload_model_gcp
from mlchartist.data import load_processed_data
from mlchartist.models import final_model, simple_model
from mlchartist.params import MODEL_VERSION
from mlchartist.array_builder import full_dataset_randomised_arrays

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import random

class Trainer(object):
    EXPERIMENT_NAME = "MLChartistModel"

    def __init__(self, unsplit_df=None, **kwargs):
        self.input_df = unsplit_df
        self.kwargs = kwargs
        self.build_numpy_arrays_kwargs = None

        self.local = kwargs.get("local", True)
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)
        ## outputs
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.scaler = None
        self.model = None


    def build_numpy_arrays(self, **kwargs):
        self.build_numpy_arrays_kwargs = kwargs
        self.train_x, self.train_y, self.test_x, self.test_y, self.scaler = full_dataset_randomised_arrays(
                                                                                    unsplit_df=self.input_df, **self.build_numpy_arrays_kwargs)

    def load_numpy_arrays(self):
        pass

    def save_numpy_arrays(self):
        pass

    def define_model(self):
        """return model"""
        ##self.model = final_model()
        self.model = simple_model()

    def fit_model(self, **kwargs):
        random_sample = kwargs.get("random_sample", True)
        if random_sample == True:
            indx = list(range(len(self.train_x)))
            sample_indx = random.sample(indx, 250000)
            X_train_sample =  self.train_x[[sample_indx], :][0]
            y_train_sample = self.train_y[[sample_indx]]
        else:
            X_train_sample = self.train_x
            y_train_sample = self.train_y

        early_stop_patience = kwargs.get("early_stop_patience", 10)
        restore_best_weights = kwargs.get("restore_best_weights", True)
        reduce_LR_patience = kwargs.get("reduce_LR_patience", 3)
        verbose = kwargs.get("verbose", 1)
        min_lr = kwargs.get("min_lr", 0.000005)
        epochs = kwargs.get("epochs", 50) ## 500
        batch_size = kwargs.get("batch_size", 16) ## 16
        validation_split = kwargs.get("validation_split", 10)
        print('early_stop_patience', early_stop_patience)

        es = EarlyStopping(patience=early_stop_patience, restore_best_weights=restore_best_weights)
        rp = ReduceLROnPlateau(patience=reduce_LR_patience, verbose=verbose, min_lr=min_lr)

        print('X_train_sample', len(X_train_sample))
        print('y_train_sample', len(y_train_sample))

        # self.model.fit(X_train_sample, y_train_sample, 
        #         epochs=epochs, 
        #         batch_size=batch_size,
        #         validation_split=validation_split,
        #         callbacks=[es, rp])

        self.model.fit(X_train_sample, y_train_sample, 
                    epochs=10, 
                    validation_split=0.1)

    def save_model(self):
        """evaluates the pipeline on df_test"""
        self.model.save('models/test_model.joblib')
        #joblib.dump(self.model, '../models/test_model.joblib')
        print("test_model.joblib saved locally")
        if not self.local:
            upload_model_gcp(model_version=MODEL_VERSION)

    def evaluate_model(self):
        print('Model Evaluation:', self.model.evaluate(self.test_x, self.test_y))

if __name__ == "__main__":
    ##warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    experiment = "mlchartist_set_Ian"
    #gcp_details = load_gcp_credentials()
    data_params = dict(nrows=10000, 
                        local=False, 
                        ticker_list=['AAPL', 'TSLA'], 
                        min_length=500, 
                        nasdaq100=False, 
                        gcp_credentials_path=None)
    print("############   Loading Processed CSVs   ############")
    df = load_processed_data(**data_params)
    print("############   Start Trainer   ############")
    model_trainer = Trainer(unsplit_df=df, upload=True, local=False, experiment_name=experiment)
    del df
    print("############   Generating Numpy Arrays   ############")
    INPUT_COLS = ['RSI', 'Stochastic', 'Stochastic_signal', 'ADI','OBV', 'ATR', 'ADX', 'ADX_pos', 'ADX_neg', 'MACD', 'MACD_diff',
              'MACD_signal', '5TD_return', '10TD_return', '20TD_return']
    TARGET_COLS=['5TD_return', '10TD_return', '20TD_return']
    outlier_validation={'5TD_return': [-0.5, 0.5]}
    stride = 50
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
    
    model_trainer.build_numpy_arrays(**build_array_params)
    print("############  Define model   ############")
    model_trainer.define_model()
    print("############  Training model   ############")
    model_trainer.fit_model(random_sample=False)
    print("############  Evaluating model ############")
    model_trainer.evaluate_model()
    print("############   Saving model    ############")
    model_trainer.save_model()
