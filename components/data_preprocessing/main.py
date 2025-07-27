
from ingestion.data_extraction import DataIngestion
from transformation.tranformation import DataTransformation
from feature_engineering.features_builder import FeatureEngineering
from logger.logging_master import logger
import time
import sys

def main():
    start_time = time.time()
    logger.info("Starting Data Preprocessing Pipeline")
    
    try:
        # Step 1: Data Ingestion
        logger.info("Step 1: Data Ingestion")
        ingestion = DataIngestion()
        ingestion.download_all()
        logger.info("✓ Data Ingestion completed")
        
        # Step 2: Data Transformation
        logger.info("Step 2: Data Transformation")
        transformation = DataTransformation()
        transformation.data_prep()
        logger.info("✓ Data Transformation completed")
        
        # Step 3: Feature Engineering
        logger.info("Step 3: Feature Engineering")
        feature_engineering = FeatureEngineering()
        feature_engineering.feature_creation()
        logger.info("✓ Feature Engineering completed")
        
        # Pipeline Summary
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__=='__main__':
    main()
    

    




