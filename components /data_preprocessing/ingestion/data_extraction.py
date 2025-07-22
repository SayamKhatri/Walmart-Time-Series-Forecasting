import pandas as pd
import boto3 
import pyarrow
from config.config_manager import ConfigManager
import os


class DataIngestion:
    def __init__(self):
        self.config = ConfigManager().get_data_ingestion_config()
        self.s3_client = boto3.client('s3')
        
    def download_file(self, s3_key, path):
        os.makedirs(self.config.download_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.download_dir, self.config.download_sub_dir), exist_ok=True)
        
        self.s3_client.download_file(self.config.bucket_name, s3_key, path)

    def download_all(self):
        self.download_file(self.config.calender_file_key, self.config.calender_dim_path)
        self.download_file(self.config.product_file_key, self.config.product_dim_path)
        self.download_file(self.config.sales_fact_key, self.config.sales_fact_path)