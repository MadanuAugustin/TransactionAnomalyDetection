




import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.TransactionAnomalyDetection.logger_file.logger_obj import logger
from src.TransactionAnomalyDetection.Exception.custom_exception import CustomException


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts//model_trainer//model.joblib'))
        self.preprocessorObj = joblib.load(Path('artifacts//data_transformation//preprocessor_obj.joblib'))


    # the below method takes the data from the user to predict

    def predictDatapoint(self, data):
        
        try:

            data_df = data.rename(columns = {0 : 'Transaction_Amount', 1 : 'Average_Transaction_Amount', 2 : 'Frequency_of_Transactions'})
            
            print(data_df)

            transformed_numeric_cols = self.preprocessorObj.transform(data_df)

            transformed_user_input = pd.DataFrame(transformed_numeric_cols)

            logger.info(f'---------Below is the transformed user input----------------')

            print(transformed_user_input)


            prediction = self.model.predict(transformed_user_input)

            print(prediction)

            list_output  = []

            if prediction == [1]:
                list_output.append('No_anomaly')
            elif prediction == [-1]:
                list_output.append('Anomaly')

            logger.info(f'-----------Below output is predicted by the model---------------')

            print(list_output)

            return list_output
        
        
        except Exception as e:
            raise CustomException(e, sys)

