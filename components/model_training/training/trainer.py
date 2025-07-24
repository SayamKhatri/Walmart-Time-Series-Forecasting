import pandas as pd 
import numpy as np
from config.config_manager import ConfigManager
import os
import lightgbm as lgb
import torch
from constants.constants import params

class ModelTraining:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_model_training_config()

    def split_data(self):
        df = pd.read_parquet(self.config.data_save_path)
        df['day_num'] = df['day_num'].astype(int) - 730

        df = df.sort_values('day_num')

        unique_days = df['day_num'].unique()
        total_days = len(unique_days)

        train_end_day = unique_days[int(0.7 * total_days)]
        valid_end_day = unique_days[int(0.85 * total_days)]

        # Split data
        train_data = df[df['day_num'] <= train_end_day]
        valid_data = df[(df['day_num'] > train_end_day) & (df['day_num'] <= valid_end_day)]
        test_data = df[df['day_num'] > valid_end_day]
        
        os.makedirs(os.path.join(self.config.save_dir, self.config.save_sub_dir), exist_ok=True)

        # train_data.to_parquet(self.config.train_split_path, index=False)
        # valid_data.to_parquet(self.config.valid_split_path, index=False)
        test_data.to_parquet(self.config.test_split_path, index=False)
        print('Data Splitted')

        self.training(train_data[:-1000], valid_data[1000:])

    def training(self, df_train, df_valid):
        X_train = df_train.drop(columns=['day_num', 'sales'])
        y_train = df_train['sales']

        X_valid = df_valid.drop(columns=['day_num', 'sales'])
        y_valid = df_valid['sales']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)


        parameters = params.copy()
        
        if torch.cuda.is_available():
            parameters.update(
                {
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                }
            )
        else:
            parameters['device'] = 'cpu'

        print('Using:', parameters['device'])
        print('Starting Training')
        # train
        model = lgb.train(
            parameters,
            train_data,
            num_boost_round=10,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=5),
                lgb.log_evaluation(period=1)
            ]
        )

        
        os.makedirs(self.config.model_artifact_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.model_artifact_path, self.config.model_save_path), exist_ok=True)
        save_path = os.path.join(self.config.model_artifact_path, self.config.model_save_path)
        
        model.save_model(os.path.join(save_path, 'lgb_model.txt'))
        print('Model Saved')

    def train_model(self):
        self.split_data()
