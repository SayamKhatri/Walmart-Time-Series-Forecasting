import pandas as pd 
from config.config_manager import ConfigManager
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
from logger.logging_master import logger
import mlflow
import mlflow.lightgbm

class ModelEvaluation:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_eval_config()
        self.s3_client = config.get_s3_client()

    def eval(self):
        logger.info("Starting model evaluation")
        
        try:
            test_data_path = os.path.join(self.config.test_split_path)
            df_test = pd.read_parquet(test_data_path)

            model_path = os.path.join(
                self.config.model_artifact_path,
                self.config.model_save_path,
                'lgb_model.txt'
            )
            model = lgb.Booster(model_file=model_path)

            X_test = df_test.drop(columns=['sales', 'day_num'])
            y_test = df_test['sales']

            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

            logger.info(f"Model evaluation completed - RMSE: {RMSE:.4f}")

            with mlflow.start_run(run_name="Model Evaluation"):
                mlflow.log_metric("RMSE", RMSE)

                if RMSE > 4.0:
                    logger.warning(f"Model performance below threshold (RMSE: {RMSE:.4f} > 4.0)")
                    mlflow.set_tag("status", "rejected")
                else:
                    logger.info(f"Model performance acceptable (RMSE: {RMSE:.4f} <= 4.0)")
                    mlflow.set_tag("status", "approved")
                    
                    # Log model to MLflow
                    mlflow.lightgbm.log_model(model, artifact_path="model")

                    self.push_model(model_path)

        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise

    def push_model(self, path):
        try:
            bucket = self.config.save_bucket_name
            key = os.path.join(self.config.save_bucket_key, 'lgb_model.txt')

            self.s3_client.upload_file(path, bucket, key)
            logger.info("Model deployed to production successfully")

        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            raise
