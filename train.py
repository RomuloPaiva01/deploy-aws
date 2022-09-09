import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

mlflow.set_experiment('iris-demo')

#load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#start training
with mlflow.start_run(run_name='my_model_experiment') as run:

    # add parameters
    num_estimators = 100
    mlflow.log_param('num_estimators', num_estimators)

    # training the model
    rf = RandomForestClassifier(n_estimators=num_estimators, random_state=0)
    rf.fit(X_train, y_train)

    # inference
    predictions = rf.predict(X_test)

    mlflow.sklearn.log_model(rf, 'random-forest-model')

    acc = accuracy_score(y_test, predictions)
    mlflow.log_metric('accuracy', acc)
    print(f'accuracy: {acc}')

    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id
    mlflow.end_run()

    print(f'run_id: {run_id}')
    print(f'artifcat_uri: {mlflow.get_artifact_uri()}')
