from dataclasses import dataclass
import os


@dataclass
class DataIngestionConfig:
    bucket_name:str = 'data-walmart-transformed'
    data_path:str = 'transformed_data'
    le_path:str = 'label_encoders'


    save_dir:str = os.path.join('data')
    download_sub_dir: str = 'raw'
    
    data_save_path:str = os.path.join(save_dir, download_sub_dir, 'full_data.parquet')
    

@dataclass
class TrainModelConfig:
    save_dir:str
    download_sub_dir: str 
    data_save_path:str
    save_sub_dir:str = 'splits'
    model_artifact_path:str = 'Artifacts'
    model_save_path: str = 'Saved_Model'
    other_artifacts_save_path:str = 'Other Artifacts'


    def __post_init__(self):
        self.train_split_path:str = os.path.join(self.save_dir, self.save_sub_dir, 'train_data.parquet')
        self.valid_split_path:str = os.path.join(self.save_dir,self.save_sub_dir, 'valid_data.parquet')
        self.test_split_path:str = os.path.join(self.save_dir, self.save_sub_dir, 'test_data.parquet')
    
@dataclass
class ModelEvaluationConfig:
    model_artifact_path:str
    model_save_path:str

    test_split_path:str 

    save_bucket_name:str = 'model-artifacts-wsf'
    save_bucket_key:str = 'Champion_Model'

