from ingestion.data_extraction import DataIngestion
from transformation.tranformation import DataTransformation
from inference.inference import Inference
from logger.logging_master import logger
import time
import sys

def main():
    start_time = time.time()
    logger.info("Starting Model Deployment Pipeline")
    
    try:
        # Step 1: Data Ingestion
        logger.info("Step 1: Data Ingestion")
        ingestion = DataIngestion()
        ingestion.download_all()
        logger.info("Data ingestion completed")
        
        # Step 2: Data Transformation
        logger.info("Step 2: Data Transformation")
        transformation = DataTransformation()
        transformation.data_prep()
        logger.info("Data transformation completed")
        
        # Step 3: Inference
        logger.info("Step 3: Model Inference")
        inference = Inference()
        inference.inference_preparation()
        logger.info("Inference completed")
        
        # Pipeline Summary
        total_time = time.time() - start_time
        logger.info(f"Model deployment pipeline completed successfully in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Model deployment pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()