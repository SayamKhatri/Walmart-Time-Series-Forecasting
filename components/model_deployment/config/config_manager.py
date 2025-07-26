from config.config import DataIngestionConfig, DataTransformationConfig, InferenceConfig
import boto3

class ConfigManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.data_ingestion_config = DataIngestionConfig()
        
        self.data_transformation_config = DataTransformationConfig(
            raw_data_path = self.data_ingestion_config.download_dir,
            raw_data_subdir_path=self.data_ingestion_config.download_sub_dir,
            le_path = self.data_ingestion_config.le_path)
        
        self.feature_engineering_config = InferenceConfig(
            self.data_transformation_config.consolidated_data_path,
            self.data_ingestion_config.le_path,
            self.data_ingestion_config.product_dim_path,
            self.data_ingestion_config.calender_dim_path
        )

    def get_data_ingestion_config(self):
        return self.data_ingestion_config
    
    def get_data_transformation_config(self):
        return self.data_transformation_config

    def get_inference_config(self):
        return self.feature_engineering_config
    
    def get_s3_client(self):
        return self.s3_client

    def get_artifact_configuration(self):
        return self.get_artifact_config