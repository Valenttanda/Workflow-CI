import mlflow 
import pandas as pd
import numpy as np
import os
import sys
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
import warnings

if __name__ == "__main__":
  warnings.filterwarnings('ignore')
  np.random.seed(50)
  
  file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_personal_finance_dataset_preprocessing.csv')
  data = pd.read_csv(file_path)
  X_train, X_test, y_train, y_test = train_test_split(
    data.drop('has_loan', axis=1),
    data['has_loan'],
    random_state=42,
    test_size=0.2
  )
  
  input_sample = X_train[0:5]
  n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
  max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 40
  
  with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    predict_quality = model.predict(X_test)
    
    mlflow.sklearn.log_model(
      sk_model=model,
      artifact_path='model',
      input_example=input_sample
    )
    
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
      
    recall = recall_score(y_test, y_pred)
    presicion = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
      
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('presicion', presicion)
    mlflow.log_metric('f1_score', f1)