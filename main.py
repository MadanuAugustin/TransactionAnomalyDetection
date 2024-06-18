

import sys
from src.TransactionAnomalyDetection.logger_file.logger_obj import logger
from src.TransactionAnomalyDetection.Exception.custom_exception import CustomException
from src.TransactionAnomalyDetection.pipeline.stage_01_dataIngestion import DataIngestionTrainingPipeline
from src.TransactionAnomalyDetection.pipeline.stage_02_dataValidation import DataValidationTrainingPipeline
from src.TransactionAnomalyDetection.pipeline.stage_03_dataTransformation import DataTransformationPipeline
from src.TransactionAnomalyDetection.pipeline.stage_04_modelTraining import ModelTrainingPipeline
from src.TransactionAnomalyDetection.pipeline.stage_05_modelEvaluation import ModelEvaluationPipeline
STAGE_NAME = 'Data Ingestion Stage'


try:
    logger.info(f'--------------------stage {STAGE_NAME} started --------------------------')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f'----------------------stage {STAGE_NAME} completed ---------------------')
except Exception as e:
    raise CustomException(e, sys)





STAGE_NAME = 'Data Validation Stage'

try:
    logger.info(f'-----------------stage {STAGE_NAME} started-----------------------')
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f'-----------------stage {STAGE_NAME} completed-----------------------')
except Exception as e:
    raise CustomException(e, sys)



STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f'-----------stage {STAGE_NAME} started------------------------')
    datatransformation = DataTransformationPipeline()
    datatransformation.main()
except Exception as e:
    raise CustomException(e, sys)


STAGE_NAME = 'Model Training Stage'

try:
    logger.info(f'--------------------stage {STAGE_NAME} started---------------------')
    model_training_obj = ModelTrainingPipeline()
    model_training_obj.main()
except Exception as e:
    raise CustomException(e, sys)




STAGE_NAME = 'Model Evaluation Stage'

try:
    logger.info(f'---------------stage {STAGE_NAME} started---------------------')
    model_evaluation_obj = ModelEvaluationPipeline()
    model_evaluation_obj.main()
except Exception as e:
    raise CustomException(e, sys)