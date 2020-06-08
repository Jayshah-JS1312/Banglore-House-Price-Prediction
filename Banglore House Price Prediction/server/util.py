import json
import pickle
import numpy as np

locations = None
data_columns = None
model = None

def predict_price(location,sqft,bhk,bath):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0],2)

def my_location_names():
    return locations

def load_saved_artifacts():
    print("Loading Saved Artifacts....")
    global data_columns
    global locations
    global model

    with open("./artifacts/columns.json",'r') as fp:
       data_columns = json.load(fp)['data_columns']
       locations = data_columns[3:]
    
    with open("./artifacts/home_price_prediction_model.pickle",'rb') as fb:
        model = pickle.load(fb)
    print("Columns and Model loading Done......")


if __name__ == '__main__':
    load_saved_artifacts()
    print(my_location_names())
    print(predict_price('1st Phase JP Nagar',1000,2,2))