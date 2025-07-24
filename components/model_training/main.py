from ingestion.data_ingestion import DataIngestion
from training.trainer import ModelTraining
from evaluation.evaluation import ModelEvaluation

ingestion = DataIngestion()
# ingestion.download_all_artifacts()
print('ALL DOWNLAODED')
# training = ModelTraining()
# training.train_model()
print('Model Trained')
evaluation = ModelEvaluation()
evaluation.eval()
print('Read Dataset')



