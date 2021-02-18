from mlchartist.array_builder import full_dataset_randomised_arrays

from google.cloud import storage


def get_process_csv_data():
    pass

def create_numpy_arrays():
    pass

class BuildArrays(object):

    def __init__(self, unsplit_df=None, **kwargs):
        self.input_df = unsplit_df
        self.split = kwargs.get("split", True)


    def build_arrays(self):
        """return estimator"""

    def save(self, local=True, scaler=True, numpy_arrays=True):
        """define pipeline here, define it as a class attribute"""

    def output(self, X_test, y_test):
        """evaluates the pipeline on df_test"""
        pass


if __name__ == "__main__":
    ##warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    experiment = "mlchartist_set_YOURNAME"
    data_params = dict(nrows=1000,
                  upload=True,
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  gridsearch=False,
                  optimize=False,
                  estimator="xgboost",
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name=experiment)
    print("############   Loading Data   ############")
    df = load_processed_data(**data_params)
    # Train and save model, locally and
    build_array_params = dict(

    )
    arrays_pipeline = BuildArrays(unsplit_df=df, **build_array_params)
    del df
    print(colored("############  Building Arrays model   ############", "red"))
    arrays_pipeline.build_arrays()
    print(colored("############  Evaluating model ############", "blue"))
    arrays_pipeline.save()
    print(colored("############   Saving model    ############", "green"))
    x, y, z, a, scaler = arrays_pipeline.output()



