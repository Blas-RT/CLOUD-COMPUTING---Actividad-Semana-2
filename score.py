import json
import pandas as pd
import joblib
from azureml.core.model import Model

def init():
    global model
    global selected_columns  # Declare selected_columns as global

    # Load the model
    model_path = Model.get_model_path('model')
    model = joblib.load(model_path)

    # Load the selected columns from the JSON file
    selected_columns_path = Model.get_model_path('selected_columns.json')  # Ensure the JSON file is registered
    with open(selected_columns_path, 'r') as f:
        selected_columns = json.load(f)

def run(raw_data):
    try:
        # Parse input data
        data = json.loads(raw_data)['data'][0]
        data = pd.DataFrame(data)

        # Drop selected columns
        data_dummies = data.drop(selected_columns, axis=1)

        # Perform prediction
        result = model.predict(data_dummies).tolist()
        return json.dumps(result)
    except Exception as e:
        return json.dumps(str(e))
