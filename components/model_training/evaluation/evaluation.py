import pandas as pd 
from config.config_manager import ConfigManager
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

class ModelEvaluation:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_eval_config()
        self.s3_client = config.get_s3_client()

    def eval(self):
        test_data_path = os.path.join(self.config.test_split_path)
        df_test = pd.read_parquet(test_data_path)
        
        model_path = os.path.join(self.config.model_artifact_path, self.config.model_save_path, 'lgb_model.txt')
        model = lgb.Booster(model_file=model_path)


        X_test = df_test.drop(columns=['sales', 'day_num'])
        y_test = df_test['sales']

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

        if RMSE > 4.0:
            print('Action Needed')
        else:
            self.push_model(model_path)

    def push_model(self, path):
        bucket = self.config.save_bucket_name
        key = os.path.join(self.config.save_bucket_key, 'lgb_model.txt')
        
        self.s3_client
        self.s3_client.upload_file(
            path, bucket, key
        )
        print('Uploaded')








