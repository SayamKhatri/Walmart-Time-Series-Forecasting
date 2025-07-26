from config.config_manager import ConfigManager
import numpy as np
import pandas as pd 
import joblib
import os
import lightgbm as lgb
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

class Inference:
    def __init__(self):
        config = ConfigManager()
        self.config = config.get_inference_config()
        self.s3_client = config.get_s3_client()

    def inference_preparation(self):
        df_sales = pd.read_parquet(self.config.transformed_data_path)
        df_current = df_sales[-365:]
        df_current.reset_index(drop=True, inplace=True)
        df_current['day_num'] = df_current['d'].str.split('_').str[1]
        df_current['day_num'] = df_current['day_num'].astype(int)
        truth_num = df_current['day_num'].max()
        
        df_product = pd.read_parquet(self.config.product_dim_path)
        df_product = df_product[df_product['store_id'].isin(['CA_1', 'CA_2', 'CA_3', 'CA_4'])]


        df_calender = pd.read_parquet(self.config.calender_dim_path)
        df_calender['date'] = pd.to_datetime(df_calender['date'])
        df_calender['day_num'] = df_calender['d'].str.split('_').str[1]
        df_calender['day_num']= df_calender['day_num'].astype(int)

        forecast_start_date = df_calender[df_calender['day_num'] == truth_num]['date'].iloc[0]
        label_encoders = self.get_label_encoders()
        model = self.get_model()

        df_predict = self.seven_day_predictions(
            forecast_start_date= forecast_start_date,
            df_calender= df_calender,
            df_product= df_product,
            label_encoders= label_encoders,
            model= model,
            df_history= df_current
        )


        parquet_buffer = BytesIO()
        df_predict.to_parquet(parquet_buffer, index=False)

        print("[14] Uploading feature data to S3...")
        self.predict_data_to_s3(parquet_buffer)
        print("[14] Upload complete.")



    def create_dummy_rows_for_prediction(self, forecast_start_date, df_calender, df_product, label_encoders):
        cal_row = df_calender[df_calender['date'] == forecast_start_date]
        cal_row = cal_row.iloc[0]
        wm_yr_wk = cal_row['wm_yr_wk']
        day_num = cal_row['day_num']

        product_map = df_product[['store_id', 'product_key']].drop_duplicates()

        price_map = df_product[['store_id', 'product_key', 'sell_price', 'wm_yr_wk']].drop_duplicates()
        price_map = price_map[price_map['wm_yr_wk'] == wm_yr_wk]

        product_with_price = price_map.merge(product_map, how='left', on=['store_id', 'product_key'])

        for field in ['wday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA']:
            product_with_price[field] = cal_row[field]
        
        product_with_price['day_num'] = day_num


        # Some Cleaning
        product_with_price['sales'] = np.nan
        product_with_price.drop(columns=['wm_yr_wk'], inplace=True)
        product_with_price['event_name_1'].fillna('No Event', inplace=True)
        product_with_price['event_type_1'].fillna('No Event', inplace=True)
        product_with_price['event_name_2'].fillna('No Event', inplace=True)
        product_with_price['event_type_2'].fillna('No Event', inplace=True)


        for col in ['store_id', 'event_type_1', 'event_name_2', 'event_type_2', 'event_name_1']:
            product_with_price[col] = label_encoders[col].transform(product_with_price[col])
        
        return product_with_price
            
    def seven_day_predictions(self, forecast_start_date, df_history, df_calender, df_product, label_encoders, model, days=7):
        prediction_results = []
        df_current = df_history.copy()

        for i in range(days):
            pred_date = pd.to_datetime(forecast_start_date) + pd.Timedelta(days=i)
            pred_date_str = pred_date.strftime('%Y-%m-%d')

            # Create dummy rows for that day
            dummy_df = self.create_dummy_rows_for_prediction(pred_date_str, df_calender, df_product, label_encoders)
            dummy_df['date'] = pred_date  

            # Stack dummy on top of current history
            combined_df = pd.concat([df_current, dummy_df], ignore_index=True)

            combined_df['id'] = (
                combined_df['product_key'].astype(str)
                + '_store_id_' 
                + combined_df['store_id'].astype(str)
            )

            combined_df = combined_df.sort_values(by=['id', 'day_num']).reset_index(drop=True)

            # Feature engineering
            LAG_DAYS = [7, 14, 28, 45, 90, 365]
            for lag in LAG_DAYS:
                combined_df[f'lag_{lag}'] = combined_df.groupby('id')['sales'].shift(lag)

            ROLLING_WINDOWS = [(1, 7), (7, 14), (14, 28), (28, 45), (45, 90)]
            for start, end in ROLLING_WINDOWS:
                size = end - start
                combined_df[f'rolling_mean_{start}_{end}'] = (
                    combined_df.groupby('id')['sales'].shift(start).rolling(size).mean()
                )
                combined_df[f'rolling_std_{start}_{end}'] = (
                    combined_df.groupby('id')['sales'].shift(start).rolling(size).std()
                )

            combined_df["pct_change_price"] = (
                combined_df.groupby("id")["sell_price"].pct_change().fillna(0)
            )

            pred_day_num = df_calender[df_calender['date'] == pred_date_str]['day_num'].iloc[0]
            pred_rows = combined_df[combined_df['day_num'] == pred_day_num].copy()

            # Predict
            feature_cols = model.feature_name()
            X_pred = pred_rows[feature_cols]
            pred_rows['sales'] = model.predict(X_pred)

            prediction_results.append(pred_rows)
            df_current = pd.concat([df_current, pred_rows], ignore_index=True)

       
        df_forecast = pd.concat(prediction_results).reset_index(drop=True)

        # Return clean forecast
        return df_forecast[['store_id', 'product_key', 'date', 'sales']]


    def get_label_encoders(self):
        label_encoders = {}
        for col in ['store_id', 'event_type_1', 'event_name_2', 'event_type_2', 'event_name_1']:
            path = os.path.join(self.config.le_path, f'le_{col}.pkl')
            le = joblib.load(path)
            label_encoders[col] = le
        
        return label_encoders

    def get_model(self):
        os.makedirs(self.config.model_download_dir, exist_ok=True)
        save_path = os.path.join(self.config.model_download_dir, 'lgb-model.txt')
        bucket = self.config.model_bucket_name
        key = self.config.model_bucket_key
        
        self.s3_client.download_file(
            bucket, key, save_path
        )
        print('Model Downlaoded')

        model = lgb.Booster(model_file=save_path)
        return model

    def predict_data_to_s3(self, parquet_buffer):
        bucket_name = self.config.prediction_bucket
        key_name = 'prediction.parquet'

        self.s3_client.put_object(
            Bucket=bucket_name,
            Key=key_name,
            Body=parquet_buffer.getvalue()
        )