from config.config import DataIngestionConfig, DataTransformationConfig

class ConfigManager:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
        self.data_transformation_config = DataTransformationConfig(
            raw_data_path = self.data_ingestion_config.download_dir,
            raw_data_subdir_path=self.data_ingestion_config.download_sub_dir)
        
        

    def get_data_ingestion_config(self):
        return self.data_ingestion_config
    

    def get_data_transformation_config(self):
        return self.data_transformation_config

