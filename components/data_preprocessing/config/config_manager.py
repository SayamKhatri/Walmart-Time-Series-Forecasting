from config.config import DataIngestionConfig, DataTransformationConfig, FeatureEngineeringConfig
import boto3
from logger.logging_master import logger

class ConfigManager:
    def __init__(self):
        try:
            # Initialize S3 client
            self.s3_client = boto3.client('s3')
            
            # Initialize data ingestion config
            self.data_ingestion_config = DataIngestionConfig()
            
            # Initialize data transformation config
            self.data_transformation_config = DataTransformationConfig(
                raw_data_path = self.data_ingestion_config.download_dir,
                raw_data_subdir_path=self.data_ingestion_config.download_sub_dir)
            
            # Initialize feature engineering config
            self.feature_engineering_config = FeatureEngineeringConfig(
                self.data_transformation_config.consolidated_data_path,
                self.data_transformation_config.save_label_encoder_dir_path
            )
            
        except Exception as e:
            logger.error(f"Config Manager initialization failed: {str(e)}")
            raise
        
    def get_data_ingestion_config(self):
        return self.data_ingestion_config
    

    def get_data_transformation_config(self):
        return self.data_transformation_config

    def get_feature_engineering_config(self):
        return self.feature_engineering_config
    
    def get_s3_client(self):
        return self.s3_client
    

