from config.config_manager import ConfigManager
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import gc 
from logger.logging_master import logger

class DataTransformation:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_data_transformation_config()
        self.path = os.path.join(self.config.raw_data_path, self.config.raw_data_subdir_path)
        self.s3_client = config.get_s3_client()

    def tranform_data(self):
        logger.info("Transforming and consolidating data")
        
        try:
            # Load data files
            df_sales = pd.read_parquet(os.path.join(self.path, 'sales_fact.parquet'), columns=['store_id', 'product_key', 'date_key', 'sales'])
            df_products = pd.read_parquet(os.path.join(self.path, 'product_dim.parquet'))
            df_calender = pd.read_parquet(os.path.join(self.path, 'calender_dim.parquet'))
            
            # Filter products for CA stores
            df_products = df_products[df_products['store_id'].isin(['CA_1', 'CA_2', 'CA_3', 'CA_4'])]

            # Create full grid
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

            # Merge calendar
            calendar_cols = ['date_key', 'wday', 'month', 'year', 'd', 'wm_yr_wk', 
                            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA']
            merged_df = merged_df.merge(df_calender[calendar_cols], on='date_key', how='left')
            merged_df.fillna('No Event', inplace=True)
            
            # Clear up some memory
            del df_sales 
            del df_calender
            del full_grid
            del full_combo
            gc.collect()

            # Get static item_id mapping
            item_map = df_products[['store_id', 'product_key', 'item_id']].drop_duplicates()
            merged_df = merged_df.merge(item_map, on=['store_id', 'product_key'], how='left')

            # Get weekly sell_price
            price_map = df_products[['store_id', 'product_key', 'wm_yr_wk', 'sell_price']].drop_duplicates()
            final_df = merged_df.merge(price_map, on=['store_id', 'product_key', 'wm_yr_wk'], how='left')

            # Save consolidated data
            save_path_data = self.config.consolidated_data_path
            os.makedirs(os.path.join(self.config.raw_data_path, self.config.save_path), exist_ok=True)

            # Label encode store_id
            le = LabelEncoder()
            final_df['store_id'] = le.fit_transform(final_df['store_id'])
            save_path = os.path.join(self.config.save_label_encoder_dir_path, f'le_store_id.pkl')
            joblib.dump(le, save_path)
            
            # Save final data
            final_df.to_parquet(save_path_data, index=False)
            logger.info(f"Consolidated data saved: {final_df.shape}")
            
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise

    def get_label_encoders(self):
        logger.info("Creating label encoders")
        
        try:
            df_calender = pd.read_parquet(os.path.join(self.path, 'calender_dim.parquet'))
            label_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
            
            df_calender['event_name_1']= df_calender['event_name_1'].fillna('No Event')
            df_calender['event_type_1']= df_calender['event_type_1'].fillna('No Event')
            df_calender['event_name_2']= df_calender['event_name_2'].fillna('No Event')
            df_calender['event_type_2']= df_calender['event_type_2'].fillna('No Event')

            os.makedirs(os.path.join(self.config.raw_data_path, self.config.save_path_label), exist_ok=True)
            
            for col in label_cols:
                le = LabelEncoder()
                df_calender[col] = le.fit_transform(df_calender[col])
                save_path = os.path.join(self.config.save_label_encoder_dir_path, f'le_{col}.pkl')
                joblib.dump(le, save_path)
            
            logger.info("Label encoders created")
            
        except Exception as e:
            logger.error(f"Label encoder creation failed: {str(e)}")
            raise

    def data_prep(self):
        try:
            self.get_label_encoders()
            self.tranform_data()
            logger.info("Data preparation completed")
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise

        




        