from config.config_manager import ConfigManager
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import os
import gc 


class FeatureEngineering:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_feature_engineering_config()
        self.s3_client = config.get_s3_client()

    def feature_creation(self):
        path = self.config.transformed_data_path
        df = pd.read_parquet(path)
        df['id'] = df['item_id'] + '_' + df['store_id'] 
        df.sort_values(by=['id', 'date_key'], inplace=True)

        # Forward and Backward Filling Prices
        df["sell_price"] = (
            df.groupby(["item_id", "store_id"])["sell_price"]
            .ffill()
            .bfill()
        )
        df.reset_index(drop=True, inplace=True)
        df['day_num'] = df['d'].str.split('_').str[1]
        df = df[df['day_num'].astype('int') > 365]

        

        # LAG Features
        LAG_DAYS = [7, 14, 28, 45, 90, 365]
        for lag in LAG_DAYS:
            df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag)

        gc.collect()
        df = df[df['day_num'].astype('int') >= 640]
        
        # ROLLING WINDOWS FOR MEAN & STD
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

        # PRICE CHANGE PERCENTAGE 
        df.sort_values(by=['id', 'date_key'], inplace=True)
        df["pct_change_price"] = df.groupby("id")["sell_price"].pct_change()
        df["pct_change_price"].fillna(0, inplace=True)

        df = df[df['day_num'].astype('int') >= 731]
        gc.collect()
        df.drop(columns=['id', 'd', 'wm_yr_wk', 'product_key', 'date_key'], inplace=True)

        categorical_cols = ["store_id", "item_id", "event_name_1", "event_type_1",
            "event_name_2", "event_type_2"]
        
        for col in categorical_cols:
            df[col] = df[col].astype("category")


        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)

        # Upload df to s3
        self.s3_client.put_object(
            Bucket=self.config.save_bucket_name,
            Key= self.config.save_bucket_key,
            Body=parquet_buffer.getvalue()
        )


        label_cols = ['event_name_1',  'event_type_1', 'event_name_2', 'event_type_2']
        for col in label_cols:
            path = os.path.join(self.config.le_path, f'le_{col}.pkl')
            key = os.path.join(self.config.save_le_key, f'le_{col}.pkl')
            self.upload_file_to_s3(path, key)

            
    def label_encode_features(self):
        label_cols = ['event_name_1',  'event_type_1', 'event_name_2', 'event_type_2']
        label_encoders = {}
        for col in label_cols:
            path = os.path.join(self.config.le_path, f'le_{col}.pkl')
            label_encoders[col] = joblib.load(path)

    

    def upload_file_to_s3(self, path, key):
        bucket = self.config.save_bucket_name
        # Upload label encoders to s3
        self.s3_client.upload_file(
            path, bucket, key
        )
        print('Uploaded')

