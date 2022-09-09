from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import json
import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

config = read_yaml('config.yaml')

APP_NAME = config['SAGEMAKER_CONFIG']['APP_NAME']

predictor = Predictor(endpoint_name=APP_NAME, serializer=CSVSerializer())

#data
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_test = pd.DataFrame(data=X_test, columns=['var1', 'var2', 'var3', 'var4'])

csv_buffer = StringIO()
X_test.to_csv(csv_buffer, index=False)
body = csv_buffer.getvalue()

#predictions
response = predictor.predict(data=body)
print(json.loads(response))

