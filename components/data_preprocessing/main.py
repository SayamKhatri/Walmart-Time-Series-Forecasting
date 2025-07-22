
from ingestion.data_extraction import DataIngestion
from transformation.tranformation import DataTransformation

if __name__=='__main__':
    # ingestion = DataIngestion()
    # ingestion.download_all()
    print("Download complete.")

    transformation = DataTransformation()
    transformation.data_prep()
    print('Saved trasnformed data')


