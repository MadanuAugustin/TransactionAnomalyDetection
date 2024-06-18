
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from src.TransactionAnomalyDetection.logger_file.logger_obj import logger
from src.TransactionAnomalyDetection.Exception.custom_exception import CustomException
from src.TransactionAnomalyDetection.pipeline.prediction_pipeline import PredictionPipeline



# initializing the flask app

app = Flask(__name__)


# route to display the home page

@app.route('/predict', methods = ['POST', 'GET'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('index.html')
    

    else : 
        try:

            Transaction_Amount = request.form.get('Transaction_Amount')
            Average_Transaction_Amount = request.form.get('Average_Transaction_Amount')
            Frequency_of_Transactions = request.form.get('Frequency_of_Transactions')

            data = [Transaction_Amount, Average_Transaction_Amount, Frequency_of_Transactions]
            
            logger.info(f'-----------Feteched data successfully from the user--------------')
            

            data = np.array(data).reshape(1, 3)

            data = pd.DataFrame(data)

            print(data)

            obj = PredictionPipeline()

            results = obj.predictDatapoint(data)

            logger.info(f'-----------Below is the final result {results}------------------')

            print(results)

            return render_template('index.html', results = str(results))


        except Exception as e:
            raise CustomException(e, sys)
        



if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True) ## http://127.0.0.1:5000

