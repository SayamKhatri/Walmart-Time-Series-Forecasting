import pandas as pd
import boto3 
import pyarrow
from config.config_manager import ConfigManager
import os
from logger.logging_master import logger

class DataIngestion:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_data_ingestion_config()
        self.s3_client = config.get_s3_client()
        
    def download_file(self, s3_key, path):
        try:
            # Create directories
            os.makedirs(self.config.download_dir, exist_ok=True)
            os.makedirs(os.path.join(self.config.download_dir, self.config.download_sub_dir), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(self.config.bucket_name, s3_key, path)
            
            # Verify file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"Downloaded file not found at {path}")
                
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {str(e)}")
            raise

    def download_all(self):
        logger.info("Downloading data files from S3")
        
        files_to_download = [
            (self.config.calender_file_key, self.config.calender_dim_path, "Calendar"),
            (self.config.product_file_key, self.config.product_dim_path, "Product"),
            (self.config.sales_fact_key, self.config.sales_fact_path, "Sales")
        ]
        
        for s3_key, path, description in files_to_download:
            self.download_file(s3_key, path)
        
        # Verify all files exist
        all_files_exist = all(os.path.exists(path) for _, path, _ in files_to_download)
        if not all_files_exist:
            missing_files = [path for _, path, _ in files_to_download if not os.path.exists(path)]
            raise FileNotFoundError(f"Missing files: {missing_files}")
        
        logger.info("All data files downloaded successfully")