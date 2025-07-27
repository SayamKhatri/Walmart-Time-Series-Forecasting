from ingestion.data_ingestion import DataIngestion
from training.trainer import ModelTraining
from evaluation.evaluation import ModelEvaluation
from logger.logging_master import logger
import time
import sys

def main():
    start_time = time.time()
    logger.info("Starting Model Training Pipeline")
    
    try:
        # Step 1: Data Ingestion
        logger.info("Step 1: Data Ingestion")
        ingestion = DataIngestion()
        ingestion.download_all_artifacts()
        logger.info("Data ingestion completed")
        
        # Step 2: Model Training
        logger.info("Step 2: Model Training")
        training = ModelTraining()
        training.train_model()
        logger.info("Model training completed")
        
        # Step 3: Model Evaluation
        logger.info("Step 3: Model Evaluation")
        evaluation = ModelEvaluation()
        evaluation.eval()
        logger.info("Model evaluation completed")
        
        # Pipeline Summary
        total_time = time.time() - start_time
        logger.info(f"Model training pipeline completed successfully in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Model training pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()



