from config.config_manager import ConfigManager
import os
from logger.logging_master import logger

class DataIngestion:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_data_config()
        self.s3_client = config.get_s3_client()
    
    def download_data(self):
        try:
            os.makedirs(self.config.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.config.save_dir, self.config.download_sub_dir), exist_ok=True)
            
            download_key = os.path.join(self.config.data_path, 'final_transformed_data.parquet')
            self.s3_client.download_file(self.config.bucket_name, download_key, self.config.data_save_path)
            
            logger.info("Training data downloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to download training data: {str(e)}")
            raise

    def download_label_encoders(self):
        try:
            os.makedirs(os.path.join(self.config.save_dir, self.config.le_path), exist_ok=True)

            label_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
            for col in label_cols:
                save_path = os.path.join(self.config.save_dir, self.config.le_path, f'le_{col}.pkl')
                download_key = os.path.join(self.config.le_path, f'le_{col}.pkl')
                self.s3_client.download_file(self.config.bucket_name, download_key, save_path)
            
            logger.info("Label encoders downloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to download label encoders: {str(e)}")
            raise

    def download_all_artifacts(self):
        logger.info("Downloading training artifacts from S3")
        self.download_data()
        self.download_label_encoders()
        logger.info("All training artifacts downloaded successfully")