from config.config_manager import ConfigManager
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import gc 

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

        # Merge sales
        merged_df = full_combo.merge(df_sales, on=['store_id', 'product_key', 'date_key'], how='left')
        merged_df['sales'] = merged_df['sales'].fillna(0).astype('int')

        # Merge calendar to get wm_yr_wk (needed before joining sell_price)
        calendar_cols = ['date_key', 'wday', 'month', 'year', 'd', 'wm_yr_wk', 
                        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA']
        merged_df = merged_df.merge(df_calender[calendar_cols], on='date_key', how='left')
        merged_df.fillna('No Event', inplace=True) # Fills event columns only
        
        del df_sales 
        del df_calender
        del full_grid
        del full_combo
        print(gc.collect())

        # Get static item_id mapping (no week info needed)
        item_map = df_products[['store_id', 'product_key', 'item_id']].drop_duplicates()
        merged_df = merged_df.merge(item_map, on=['store_id', 'product_key'], how='left')

        # Get weekly sell_price
        price_map = df_products[['store_id', 'product_key', 'wm_yr_wk', 'sell_price']].drop_duplicates()
        final_df = merged_df.merge(price_map, on=['store_id', 'product_key', 'wm_yr_wk'], how='left')

        save_path_data = self.config.consolidated_data_path
        os.makedirs(os.path.join(self.config.raw_data_path, self.config.save_path), exist_ok=True)
        
        label_encoders = self.get_label_encoders()
        for col in ['store_id', 'event_type_1', 'event_name_2', 'event_type_2', 'event_name_1']:
            final_df[col] = label_encoders[col].transform(final_df[col])

        final_df = final_df[-365:]
        print(final_df.shape)
        print(final_df.head())
        final_df.to_parquet(save_path_data, index=False)
    
    def get_label_encoders(self):
        label_encoders = {}
        for col in ['store_id', 'event_type_1', 'event_name_2', 'event_type_2', 'event_name_1']:
            path = os.path.join(self.config.le_path, f'le_{col}.pkl')
            le = joblib.load(path)
            label_encoders[col] = le
        
        return label_encoders
    
    def data_prep(self):
        self.get_label_encoders()
        self.tranform_data()

        




        