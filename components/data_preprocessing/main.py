
from ingestion.data_extraction import DataIngestion
from transformation.tranformation import DataTransformation
from feature_engineering.features_builder import FeatureEngineering

if __name__=='__main__':
    # ingestion = DataIngestion()
    # ingestion.download_all()
    # print("Download complete.")

    transformation = DataTransformation()
    transformation.data_prep()
    print('Saved trasnformed data')

    # feature_engineering = FeatureEngineering()
    # feature_engineering.feature_creation()
    
    print('Printed first 5 rows.')
    




