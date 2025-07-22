from dataclasses import dataclass
import os

@dataclass
class DataIngestionConfig:
    bucket_name: str = 'data-walmart-raw'
    calender_file_key: str = 'calendar.parquet'
    product_file_key: str = 'products_dim.parquet'
    sales_fact_key: str = 'Real_sales_fact.parquet'


    download_dir:str = os.path.join('data')
    download_sub_dir:str = os.path.join('raw_data')

    calender_dim_path: str = os.path.join(download_dir, download_sub_dir, 'calender_dim.parquet')
    product_dim_path: str = os.path.join(download_dir, download_sub_dir, 'product_dim.parquet')
    sales_fact_path: str = os.path.join(download_dir, download_sub_dir, 'sales_fact.parquet')


@dataclass
class DataTransformationConfig:
    raw_data_path:str 
    raw_data_subdir_path:str

    save_path:str = 'transformed_data'
    save_path_label:str = 'label_encoders'

    save_bucket_name:str = 'data-walmart-transformed'
    save_bucket_key:str = 'final_transformed_data.parquet'

    def __post_init__(self):
        self.save_label_encoder_dir_path:str = os.path.join(self.raw_data_path, self.save_path_label)
        self.consolidated_data_path:str = os.path.join(self.raw_data_path, self.save_path, 'consolidated_data.parquet')

    


    




