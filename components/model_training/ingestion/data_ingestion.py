from config.config_manager import ConfigManager
import os

class DataIngestion:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_data_config()
        self.s3_client = config.get_s3_client()
    
    def download_data(self):
        os.makedirs(self.config.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, self.config.download_sub_dir), exist_ok=True)
        print(f"Downloading from bucket: {self.config.bucket_name}, key: {self.config.data_path}")
        download_key = os.path.join(self.config.data_path, 'final_transformed_data.parquet')
        self.s3_client.download_file(self.config.bucket_name, download_key, self.config.data_save_path)
        
        print('Data Downloaded')

    def download_label_encoders(self):
        os.makedirs(os.path.join(self.config.save_dir, self.config.le_path), exist_ok=True)

        label_cols = ['event_name_1',  'event_type_1', 'event_name_2', 'event_type_2']
        for col in label_cols:
            save_path = os.path.join(self.config.save_dir, self.config.le_path, f'le_{col}.pkl')
            download_key = os.path.join(self.config.le_path, f'le_{col}.pkl')
            self.s3_client.download_file(self.config.bucket_name, download_key, save_path)
            print('Downloaded', f'le_{col}.pkl')

    def download_all_artifacts(self):
        self.download_data()
        self.download_label_encoders()