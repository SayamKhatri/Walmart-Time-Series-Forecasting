from ingestion.data_extraction import DataIngestion
from transformation.tranformation import DataTransformation
from inference.inference import Inference

# ingestion = DataIngestion()
# ingestion.download_all()
# print("Download complete.")

# transformation = DataTransformation()
# transformation.data_prep()
# print('Saved trasnformed data')

inference = Inference()
inference.inference_preparation()