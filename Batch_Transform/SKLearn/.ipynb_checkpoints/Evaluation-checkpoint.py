def model_fn(model_dir):
    import os
    import pandas
    import logging
    import argparse
    import joblib


    ## Creating a logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info("Inference started.")
    
    
    
    print("Files are")
    dir_list = os.listdir("/opt/ml/model")
    print(dir_list)
    
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor







def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    import pandas
    from io import StringIO
    
    print("Request body")
    print(request_body)
    print(type(request_body))
    print("String IO request body")
    # print(StringIO(request_body.decode("utf-8"))) # Batch
    print(StringIO(request_body)) # Endpoint
    
    # df = pandas.read_csv(StringIO(request_body.decode("utf-8"))) # Batch
    if request_body[0] == 'A':
        df = pandas.read_csv(StringIO(request_body))
    else:
        df = pandas.read_csv(StringIO(request_body), header = None) # Endpoint
        # print(len(df.columns.tolist()))
        
        df.columns = ["Account length", "Number vmail messages", "Total day minutes", "Total day calls", "Total eve minutes", 
                      "Total eve calls", "Total night minutes", "Total night calls", "Total intl minutes", "Total intl calls",
                      "Customer service calls", "Total_minutes", "Total_calls", "Minutes_per_call_overall", "Minutes*call_overall",
                      "Minutes_per_call_int", "Minutes*call_int", "Minutes_per_call_day", "Minutes*call_day", "Minutes_per_call_eve",
                      "Minutes*call_eve", "Minutes_per_call_night", "Minutes*call_night", "Total_charge", 
                      "Day_minutes_per_customer_service_calls", "Day_minutes*customer_service_calls", "Total_day_minutes_wholenum",
                      "Total_day_minutes_decimalnum", "Total_minutes_wholenum", "Total_minutes_decimalnum", "Voice_and_Int_plan",
                      "Only_Int_plan", "Only_vmail_plan", "No_plans", "State_AL", "State_AR", "State_AZ", "State_CA", "State_CO",
                      "State_CT", "State_DC", "State_DE", "State_FL", "State_GA", "State_HI", "State_IA", "State_ID", "State_IL", 
                      "State_IN", "State_KS", "State_KY", "State_LA", "State_MA", "State_MD", "State_ME", "State_MI", "State_MN",
                      "State_MO", "State_MS", "State_MT", "State_NC", "State_ND", "State_NE", "State_NH", "State_NJ", "State_NM", 
                      "State_NV", "State_NY", "State_OH", "State_OK", "State_OR", "State_PA", "State_RI", "State_SC", "State_SD", 
                      "State_TN", "State_TX", "State_UT", "State_VA", "State_VT", "State_WA", "State_WI", "State_WV", "State_WY",
                      "Area code_415", "Area code_510", "International plan_Yes", "Voice mail plan_Yes", "Account_length_bins_q2",
                      "Account_length_bins_q3", "Account_length_bins_q4", "zero_vmails_Yes", "Customer_service_calls_bins_q2", 
                      "Customer_service_calls_bins_q3", "Customer_service_calls_bins_q4"]
    print(df.head())
    return df
#     if request_content_type == "text/csv":
#         df = pandas.read_csv(StringIO(request_body))
#         return df.drop(columns = ["Churn"])
#     else:
#         # Handle other content-types here or raise an Exception
#         # if the content type is not supported.
#         raise ValueError("{} not supported by script!".format(request_content_type))
    
    

    

# def output_fn(prediction, accept):
#     """Format prediction output

#     The default accept/content-type between containers for serial inference is JSON.
#     We also want to set the ContentType or mimetype as the same value as accept so the next
#     container can read the response payload correctly.
#     """
#     if accept == "application/json":
#         instances = []
#         for row in prediction.tolist():
#             instances.append({"features": row})

#         json_output = {"instances": instances}

#         return worker.Response(json.dumps(json_output), accept, mimetype=accept)
#     elif accept == 'text/csv':
#         return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
#     else:
#         raise RuntimeException("{} accept type is not supported by this script.".format(accept))
        
        





def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    # import xgboost as xgb
    from sklearn.metrics import accuracy_score
    import pathlib
    import json
    import os
    import pandas
    import numpy
    import pathlib
    
    # print(f"Input data columns: {input_data.columns}")
    # print(f"Model columns: {model.feature_names}")
    # features = model.predict(xgb.DMatrix(input_data.values))
    # columns = input_data.columns.tolist()
    # x_columns = columns.pop(columns.index("Churn"))
    # prediction_array = model.predict(input_data.loc[:,x_columns])
    print("Printing")
    print(input_data)
    print("Printing type")
    print(type(input_data))
    
    prediction_probabilities = model.predict_proba(input_data)
    print(prediction_probabilities)
    print(prediction_probabilities.shape)
    prediction_array = model.predict(input_data)
    print(prediction_array)
    print(prediction_array.shape)
    constant_array = numpy.full(prediction_array.shape, 1)
    # prediction_dataframe = pandas.DataFrame(numpy.vstack((prediction_array, prediction_probabilities)).T)
    prediction_dataframe = pandas.DataFrame(numpy.vstack((prediction_array, constant_array)).T)
    
#     prediction_dataframe = input_data.copy()
#     prediction_dataframe["Predicted_Churn"] = prediction_array
    
#     all_columns = prediction_dataframe.columns.tolist()
#     all_columns.insert(0, all_columns.pop(all_columns.index("Predicted_Churn")))
    
#     prediction_dataframe = prediction_dataframe.loc[:, all_columns]
    
    
    # accuracy = accuracy_score(input_data.Churn, features)
    # model_data_s3_location = os.environ["MODELS3LOCATION"]
    # model_name = os.environ["MODELNAME"]
    
#     report_dict = {
#         "metrics": {
#             "accuracy": {
#                 "value": accuracy,
#                 # "standard_deviation": std
#             },
#         },
#         # "model_data":model_data_s3_location,
#         # "model_name":model_name
#     }
    
#     output_dir = "/opt/ml/processing/evaluation"
#     pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     evaluation_path = f"{output_dir}/evaluation.json"
#     with open(evaluation_path, "w") as f:
#         f.write(json.dumps(report_dict))
    
    # prediction_dataframe.to_csv("/opt/ml/processing/evaluation/Prediction.csv", index = False)
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # prediction_dataframe.to_csv(f"{output_dir}/Prediction.csv", index = False) #2
    
    prediction_list = prediction_array.tolist()
    print("Done")
    
    return {"predictions":prediction_list} #1
    # return report_dict

#     if label_column in input_data:
#         # Return the label (as the first column) and the set of features.
#         return np.insert(features, 0, input_data[label_column], axis=1)
#     else:
#         # Return only the set of features
#         return features


# def output_fn(predictions, response_content_type):
#     return json.dumps(predictions)

    
    
    

# if __name__ =='__main__':
#     model_fn(model_dir = "")
