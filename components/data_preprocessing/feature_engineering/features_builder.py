from config.config_manager import ConfigManager
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import os
import gc
import time


class FeatureEngineering:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_feature_engineering_config()
        self.s3_client = config.get_s3_client()

    def feature_creation(self):
        start_time = time.time()
        print("[1] Reading transformed data...")
        path = self.config.transformed_data_path
        df = pd.read_parquet(path)
        print(f"[1] Done. Shape: {df.shape}")

        print("[2] Creating ID column and sorting...")
        df['id'] = df['item_id'] + 'store_id_' + df['store_id'].astype(str)
        df.sort_values(by=['id', 'date_key'], inplace=True)

        print("[3] Forward and backward filling 'sell_price'...")
        df["sell_price"] = (
            df.groupby(["item_id", "store_id"])["sell_price"]
            .ffill()
            .bfill()
        )
        df.reset_index(drop=True, inplace=True)

        print("[4] Filtering rows: day_num > 365...")
        df['day_num'] = df['d'].str.split('_').str[1]
        df = df[df['day_num'].astype('int') > 365]
        print(f"[4] Done. Shape after filter: {df.shape}")

        print("[5] Creating lag features...")
        LAG_DAYS = [7, 14, 28, 45, 90, 365]
        for lag in LAG_DAYS:
            df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
        print("[5] Lag features created.")

        gc.collect()
        print("[6] Filtering rows: day_num >= 640...")
        df = df[df['day_num'].astype('int') >= 640]
        print(f"[6] Done. Shape after filter: {df.shape}")

        print("[7] Creating rolling features...")
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
        print("[7] Rolling mean and std features created.")

        print("[8] Calculating percent change in 'sell_price'...")
        df.sort_values(by=['id', 'date_key'], inplace=True)
        df["pct_change_price"] = df.groupby("id")["sell_price"].pct_change()
        df["pct_change_price"].fillna(0, inplace=True)

        print("[9] Final filtering: day_num >= 731...")
        df = df[df['day_num'].astype('int') >= 731]
        print(f"[9] Done. Shape after filter: {df.shape}")

        print("[10] Dropping unnecessary columns...")
        df.drop(columns=['id', 'd', 'wm_yr_wk', 'item_id', 'date_key'], inplace=True)
        print("[10] Columns dropped.")

        print("[11] Label encoding categorical features...")
        categorical_cols = ["event_name_1", "event_type_1",
                            "event_name_2", "event_type_2"]
        label_encoders = self.label_encode_features(categorical_cols)
        for col in categorical_cols:
            df[col] = label_encoders[col].transform(df[col])

        print("[13] Saving to in-memory parquet buffer...")
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)

        print("[14] Uploading feature data to S3...")
        self.s3_client.put_object(
            Bucket=self.config.save_bucket_name,
            Key=self.config.save_bucket_key,
            Body=parquet_buffer.getvalue()
        )
        print("[14] Upload complete.")

        print("[15] Uploading label encoders to S3...")
        label_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        for col in label_cols:
            path = os.path.join(self.config.le_path, f'le_{col}.pkl')
            key = os.path.join(self.config.save_le_key, f'le_{col}.pkl')
            self.upload_file_to_s3(path, key)
        print("[15] Label encoders uploaded.")

        print(f"[DONE] Feature engineering complete. Final shape: {df.shape}")
        print(f"[TIME] Total time: {round(time.time() - start_time, 2)} seconds")

    def label_encode_features(self, label_cols):
        label_encoders = {}
        for col in label_cols:
            path = os.path.join(self.config.le_path, f'le_{col}.pkl')
            label_encoders[col] = joblib.load(path)
        return label_encoders

    def upload_file_to_s3(self, path, key):
        bucket = self.config.save_bucket_name
        self.s3_client.upload_file(path, bucket, key)
        print(f"[UPLOAD] {key} uploaded to {bucket}")
