from config.config_manager import ConfigManager
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import os
import gc
import time
from logger.logging_master import logger
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_feature_engineering_config()
        self.s3_client = config.get_s3_client()

    def feature_creation(self):
        start_time = time.time()
        logger.info("Starting feature engineering")
        
        try:
            path = self.config.transformed_data_path
            df = pd.read_parquet(path)
            logger.info(f"Loaded data: {df.shape}")

            df['id'] = df['item_id'] + 'store_id_' + df['store_id'].astype(str)
            df.sort_values(by=['id', 'date_key'], inplace=True)
        
            df["sell_price"] = (
                df.groupby(["item_id", "store_id"])["sell_price"]
                .ffill()
                .bfill()
            )
            df.reset_index(drop=True, inplace=True)

            df['day_num'] = df['d'].str.split('_').str[1]
            df = df[df['day_num'].astype('int') > 365]

            # Create lag features
            LAG_DAYS = [7, 14, 28, 45, 90, 365]
            for lag in LAG_DAYS:
                df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag)

            # Memory cleanup
            gc.collect()

            df = df[df['day_num'].astype('int') >= 640]

            # Create rolling features
            ROLLING_WINDOWS = [
                (1, 7), 
                (7, 14),
                (14, 28),
                (28, 45),
                (45, 90)
            ]
            for start, end in ROLLING_WINDOWS:
                window_size = end - start
                df[f'rolling_mean_{start}_{end}'] = (
                    df.groupby('id')['sales']
                    .shift(start)
                    .rolling(window_size)
                    .mean()
                    .round(2)
                )

                df[f'rolling_std_{start}_{end}'] = (
                    df.groupby('id')['sales']
                    .shift(start)
                    .rolling(window=window_size)
                    .std()
                    .round(2)
                )

            # Price change features
            df.sort_values(by=['id', 'date_key'], inplace=True)
            df["pct_change_price"] = df.groupby("id")["sell_price"].pct_change()
            df["pct_change_price"].fillna(0, inplace=True)

            df = df[df['day_num'].astype('int') >= 731]

            df.drop(columns=['id', 'd', 'wm_yr_wk', 'item_id', 'date_key'], inplace=True)

            categorical_cols = ["event_name_1", "event_type_1",
                                "event_name_2", "event_type_2"]
            label_encoders = self.label_encode_features(categorical_cols)
            for col in categorical_cols:
                df[col] = label_encoders[col].transform(df[col])

            parquet_buffer = BytesIO()
            df.to_parquet(parquet_buffer, index=False)

            self.s3_client.put_object(
                Bucket=self.config.save_bucket_name,
                Key=self.config.save_bucket_key,
                Body=parquet_buffer.getvalue()
            )

            label_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'store_id'] 
            for col in label_cols:
                path = os.path.join(self.config.le_path, f'le_{col}.pkl')
                key = os.path.join(self.config.save_le_key, f'le_{col}.pkl')
                self.upload_file_to_s3(path, key)

            total_time = time.time() - start_time
            logger.info(f"Feature engineering completed: {df.shape} in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

    def label_encode_features(self, label_cols):
        label_encoders = {}
        for col in label_cols:
            path = os.path.join(self.config.le_path, f'le_{col}.pkl')
            label_encoders[col] = joblib.load(path)
        return label_encoders

    def upload_file_to_s3(self, path, key):
        try:
            bucket = self.config.save_bucket_name
            self.s3_client.upload_file(path, bucket, key)
        except Exception as e:
            logger.error(f"Failed to upload {key}: {str(e)}")
            raise
