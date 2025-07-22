
from ingestion.data_extraction import DataIngestion
from transformation.tranformation import DataTransformation

if __name__=='__main__':
    # ingestion = DataIngestion()
    # ingestion.download_all()
    print("Download complete.")

    transformation = DataTransformation()
    transformation.tranform_data()
    print('Printed df_sales first 5 rows')

