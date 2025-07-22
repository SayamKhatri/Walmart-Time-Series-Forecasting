from config.config_manager import ConfigManager
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import joblib

class DataTransformation:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_data_transformation_config()
        self.path = os.path.join(self.config.raw_data_path, self.config.raw_data_subdir_path)
        self.s3_client = config.get_s3_client()

    def tranform_data(self):
        df_sales = pd.read_parquet(os.path.join(self.path, 'sales_fact.parquet'), columns=['store_id', 'product_key', 'date_key', 'sales'])
        df_products = pd.read_parquet(os.path.join(self.path, 'product_dim.parquet'))
        df_calender = pd.read_parquet(os.path.join(self.path, 'calender_dim.parquet'))

        df_products = df_products[df_products['store_id'].isin(['CA_1', 'CA_2', 'CA_3', 'CA_4'])]
        
        unique_dates = df_sales['date_key'].unique()
        product_map = df_products[['store_id', 'product_key']].drop_duplicates()


        # Cartesian product
        full_grid = pd.MultiIndex.from_product(
            [product_map['store_id'].unique(), unique_dates],
            names=['store_id', 'date_key']
            ).to_frame(index=False)

        full_combo = full_grid.merge(product_map, on='store_id', how='left')

        merged_df = full_combo.merge(df_sales, on=['store_id', 'product_key', 'date_key'], how='left')
        merged_df['sales'] = merged_df['sales'].fillna(0).astype('int')
        
        caledner_cols = ['date_key', 'wday', 'month', 'year', 'd', 'event_name_1', 
                         'event_type_1', 'event_name_2', 'event_type_2','snap_CA']
        
        final_df = merged_df.merge(
            df_calender[caledner_cols],
            on='date_key',
            how='left'
        )
        save_path = self.config.consolidated_data_path
        os.makedirs(os.path.join(self.config.raw_data_path, self.config.save_path), exist_ok=True)
        final_df.to_parquet(save_path, index=False)

        # parquet_buffer = BytesIO()
        # final_df.to_parquet(parquet_buffer, index=False)

        # # Upload to s3
        # self.s3_client.put_object(
        #     Bucket=self.config.save_bucket_name,
        #     Key= self.config.save_bucket_key,
        #     Body=parquet_buffer.get_value()
        # )

    def get_label_encoders(self):
        df_calender = pd.read_parquet(os.path.join(self.path, 'calender_dim.parquet'))
        label_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        os.makedirs(os.path.join(self.config.raw_data_path, self.config.save_path_label), exist_ok=True)
        for col in label_cols:
            le = LabelEncoder()
            df_calender[col] = le.fit_transform(df_calender[col])
            save_path = os.path.join(self.config.save_label_encoder_dir_path, f'le_{col}.pkl')
            joblib.dump(le,save_path)


    def data_prep(self):
        self.tranform_data()
        self.get_label_encoders()
        




        