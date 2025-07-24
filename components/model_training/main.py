from ingestion.data_ingestion import DataIngestion
from training.trainer import ModelTraining

ingestion = DataIngestion()
# ingestion.download_all_artifacts()
print('ALL DOWNLAODED')
training = ModelTraining()
# training.train_model()
print('Model Trained')



