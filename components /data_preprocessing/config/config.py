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

    def __post__init__(self):
        save_label_encoder_dir:str = os.path.join(self.raw_data_path, self.save_path_label)
        consolidated_data:str = os.path.join(self.raw_data_path, self.save_path, 'consolidated_data.parquet')


    




