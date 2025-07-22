from config.config_manager import ConfigManager
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

class DataTransformation:
    def __init__(self):
        self.config = ConfigManager().get_data_transformation_config()
        self.path = os.path.join(self.config.raw_data_path, self.config.raw_data_subdir_path)

    def tranform_data(self):
        df_sales = pd.read_parquet(os.path.join(self.path, 'sales_fact.parquet'))
        

    def get_label_encoders(self):
        pass

        