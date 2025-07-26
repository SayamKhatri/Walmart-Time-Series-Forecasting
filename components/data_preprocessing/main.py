
from ingestion.data_extraction import DataIngestion
from transformation.tranformation import DataTransformation
from feature_engineering.features_builder import FeatureEngineering
from logger.logging_master import logger
import time
import sys

def main():
    start_time = time.time()
    logger.info("=" * 50)
    logger.info("Starting Data Preprocessing Pipeline")
    logger.info("=" * 50)
    
    try:
        # Step 1: Data Ingestion
        logger.info("Step 1: Starting Data Ingestion...")
        ingestion_start = time.time()
        
        ingestion = DataIngestion()
        ingestion.download_all()
        
        ingestion_time = time.time() - ingestion_start
        logger.info(f"✓ Data Ingestion completed successfully in {ingestion_time:.2f} seconds")
        
        # Step 2: Data Transformation
        logger.info("Step 2: Starting Data Transformation...")
        transformation_start = time.time()
        
        transformation = DataTransformation()
        transformation.data_prep()
        
        transformation_time = time.time() - transformation_start
        logger.info(f"✓ Data Transformation completed successfully in {transformation_time:.2f} seconds")
        
        # Step 3: Feature Engineering
        logger.info("Step 3: Starting Feature Engineering...")
        feature_start = time.time()
        
        feature_engineering = FeatureEngineering()
        feature_engineering.feature_creation()
        
        feature_time = time.time() - feature_start
        logger.info(f"✓ Feature Engineering completed successfully in {feature_time:.2f} seconds")
        
        # Pipeline Summary
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("Data Preprocessing Pipeline Completed Successfully!")
        logger.info(f"Total Pipeline Time: {total_time:.2f} seconds")
        logger.info(f"Breakdown:")
        logger.info(f"  - Data Ingestion: {ingestion_time:.2f}s ({ingestion_time/total_time*100:.1f}%)")
        logger.info(f"  - Data Transformation: {transformation_time:.2f}s ({transformation_time/total_time*100:.1f}%)")
        logger.info(f"  - Feature Engineering: {feature_time:.2f}s ({feature_time/total_time*100:.1f}%)")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Pipeline terminated unsuccessfully")
        sys.exit(1)

if __name__=='__main__':
    main()
    

    




