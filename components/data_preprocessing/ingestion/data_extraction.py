import pandas as pd
import boto3 
import pyarrow
from config.config_manager import ConfigManager
import os
from logger.logging_master import logger
import time

class DataIngestion:
    def __init__(self):
        logger.info("Initializing Data Ingestion component...")
        config = ConfigManager()
        self.config = config.get_data_ingestion_config()
        self.s3_client = config.get_s3_client()
        logger.info(f"✓ Data Ingestion initialized successfully")
        logger.info(f"  - S3 Bucket: {self.config.bucket_name}")
        logger.info(f"  - Download Directory: {self.config.download_dir}")
        
    def download_file(self, s3_key, path):
        logger.info(f"Downloading file: {s3_key}")
        start_time = time.time()
        
        try:
            # Create directories
            os.makedirs(self.config.download_dir, exist_ok=True)
            os.makedirs(os.path.join(self.config.download_dir, self.config.download_sub_dir), exist_ok=True)
            logger.debug(f"✓ Directories created/verified: {self.config.download_dir}")
            
            # Download file
            self.s3_client.download_file(self.config.bucket_name, s3_key, path)
            
            # Verify file exists and get size
            if os.path.exists(path):
                file_size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
                download_time = time.time() - start_time
                logger.info(f"✓ Downloaded {s3_key} successfully")
                logger.info(f"  - File size: {file_size:.2f} MB")
                logger.info(f"  - Download time: {download_time:.2f} seconds")
                logger.info(f"  - Speed: {file_size/download_time:.2f} MB/s")
            else:
                raise FileNotFoundError(f"Downloaded file not found at {path}")
                
        except Exception as e:
            logger.error(f"✗ Failed to download {s3_key}: {str(e)}")
            raise

    def download_all(self):
        logger.info("Starting download of all required files...")
        total_start_time = time.time()
        
        files_to_download = [
            (self.config.calender_file_key, self.config.calender_dim_path, "Calendar Dimension"),
            (self.config.product_file_key, self.config.product_dim_path, "Product Dimension"),
            (self.config.sales_fact_key, self.config.sales_fact_path, "Sales Fact")
        ]
        
        for s3_key, path, description in files_to_download:
            logger.info(f"Downloading {description}...")
            self.download_file(s3_key, path)
        
        total_time = time.time() - total_start_time
        logger.info(f"✓ All files downloaded successfully in {total_time:.2f} seconds")
        
        # Verify all files exist
        all_files_exist = all(os.path.exists(path) for _, path, _ in files_to_download)
        if all_files_exist:
            logger.info("✓ All downloaded files verified successfully")
        else:
            missing_files = [path for _, path, _ in files_to_download if not os.path.exists(path)]
            logger.error(f"✗ Missing files: {missing_files}")
            raise FileNotFoundError(f"Some downloaded files are missing: {missing_files}")