from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
#import xgboost as xgb

# Load the model
#model = xgb.XGBClassifier()
model = load_model('ECommerce_ANN.h5')
#model.load_model("xgboost_model.model")
scaler = joblib.load('scaler.pkl')
app = Flask(__name__)
@app.route('/')
def loadpage():
    return render_template("index.html")
    
@app.route('/y_predict', methods = ["POST"])
def prediction():
    
    Warehouse = request.form["Warehouse"]
    Mode_of_Shipment = request.form["M_O_S"]
    Product_Importance = request.form["Importance"]
    Weight_in_Grams = request.form["Weight"]
    Customer_care_calls = request.form["Customer_care_calls"]
    x_test = {'Warehouse_block':Warehouse,
              'Mode_of_Shipment':Mode_of_Shipment,
              'Customer_care_calls':float(Customer_care_calls),
              'Product_importance':Product_Importance, 
              'Weight_in_gms':float(Weight_in_Grams)
              }
    print(f"{x_test} before encoding")
    x_test_df=pd.DataFrame([x_test])
    x_test_df['Warehouse_block'] = x_test_df['Warehouse_block'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4})
    x_test_df['Mode_of_Shipment'] = x_test_df['Mode_of_Shipment'].replace({'Flight': 0, 'Road': 1, 'Ship': 2})
    x_test_df['Product_importance'] = x_test_df['Product_importance'].replace({'low': 1, 'medium': 2, 'high': 0})
    print(f"{x_test_df} after encode")
    p = scaler.transform(x_test_df)
    #p = pd.DataFrame(p)
    #p = p.astype(np.float32)
    
    prediction = model.predict(p)
    print(prediction)
    print(type(prediction))
    prediction = prediction > 0.5
    
    if (prediction == True):
        text = "Product may not arrive on time"
    else:
        text = "Product is likely to arrive on time!"

   
    return render_template("index.html",prediction_text = text )

    
    
if __name__ == "__main__":
    app.run(debug = False)
