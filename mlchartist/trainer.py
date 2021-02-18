

class Trainer(object):

    def __init__(self, unsplit_df=None, **kwargs):
        self.pipeline = None
        self.input_df = unsplit_df
        self.split = kwargs.get("split", True)


    def get_estimator(self):
        """return estimator"""

    def set_pipeline(self):
        """define pipeline here, define it as a class attribute"""

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test"""

    def train(self):
        pass

if __name__ == "__main__":
    ##warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    experiment = "taxifare_set_YOURNAME"
    params = dict(nrows=1000,
                  upload=True,
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  gridsearch=False,
                  optimize=False,
                  estimator="xgboost",
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name=experiment)
    print("############   Loading Data   ############")
    df = get_data(**params)
    df = clean_df(df)
    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)
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
