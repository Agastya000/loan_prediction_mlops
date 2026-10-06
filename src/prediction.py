import joblib
import pandas as pd
 
model = joblib.load("./MLOps_loan_default/model/loan_default.pkl")
 
# create a test sample data
test_df = pd.read_csv("./MLOps_loan_default/data/x_test_sample.csv")
test_df.drop('Unnamed: 0', axis = 1, inplace = True)
 
print(test_df.head())
print(test_df.columns)
 
x_test_sample = test_df.sample(1)
 
yhat = model.predict(x_test_sample)
 
print(f"Prediction: {yhat}")
 