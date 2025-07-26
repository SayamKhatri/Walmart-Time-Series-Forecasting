import os 
from dataclasses import dataclass

from dataclasses import dataclass
import os

@dataclass
class DataIngestionConfig:
    bucket_name: str = 'data-walmart-raw'
    calender_file_key: str = 'calendar.parquet'
    product_file_key: str = 'products_dim.parquet'
    sales_fact_key: str = 'Real_sales_fact.parquet'

    le_bucket_name:str = 'data-walmart-transformed/label_encoders/'
    
    download_dir:str = 'data'
    download_sub_dir:str = 'raw_data'
    label_sub_dir:str = 'label_encoders'

    calender_dim_path: str = os.path.join(download_dir, download_sub_dir, 'calender_dim.parquet')
    product_dim_path: str = os.path.join(download_dir, download_sub_dir, 'product_dim.parquet')
    sales_fact_path: str = os.path.join(download_dir, download_sub_dir, 'sales_fact.parquet')
    le_path: str = os.path.join(download_dir, label_sub_dir)


@dataclass
class DataTransformationConfig:
    raw_data_path:str 
    raw_data_subdir_path:str
    le_path:str
    
    save_path:str = 'transformed_data'
    save_path_label:str = 'label_encoders'

    def __post_init__(self):
        self.save_label_encoder_dir_path:str = os.path.join(self.raw_data_path, self.save_path_label)
        self.consolidated_data_path:str = os.path.join(self.raw_data_path, self.save_path, 'consolidated_data.parquet')

    
@dataclass
class InferenceConfig:
    transformed_data_path:str
    le_path:str 
    product_dim_path:str 
    calender_dim_path: str

    model_bucket_name:str = 'model-artifacts-wsf'
    model_bucket_key:str = os.path.join('Champion_Model', 'lgb_model.txt')

    model_download_dir:str = 'model'

    prediction_bucket:str = 'wsf-predictions'
    


