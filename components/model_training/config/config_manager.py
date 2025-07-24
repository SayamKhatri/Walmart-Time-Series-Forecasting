from config.config_entities import DataIngestionConfig, TrainModelConfig
import boto3

class ConfigManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.get_data_ingestion_config = DataIngestionConfig()

        self.get_train_config = TrainModelConfig(
            self.get_data_ingestion_config.save_dir,
            self.get_data_ingestion_config.download_sub_dir,
            self.get_data_ingestion_config.data_save_path
        )


    def get_data_config(self):
        return self.get_data_ingestion_config
    

    def get_s3_client(self):
        return self.s3_client

    def get_model_training_config(self):
        return self.get_train_config