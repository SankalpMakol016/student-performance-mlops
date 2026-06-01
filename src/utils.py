import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train,Y_train,X_test,Y_test,models,params):
    try:
        report = {}
        
        best_params_report={}
        for model_name,model in models.items():
            
            para = params[model_name]
            
            gs = GridSearchCV(
                estimator=model,param_grid= para,cv=3,n_jobs=1,verbose=0
            )
            
            gs.fit(X_train,Y_train)
            
            best_params_report[model_name] = gs.best_params_
            
            model.set_params(**gs.best_params_)
            
            model.fit(X_train,Y_train)
            
            y_test_pred = model.predict(X_test)
            
            test_model_score = r2_score(y_test_pred,Y_test)
            
            report[model_name] = test_model_score
            
            return report ,best_params_report
        
    except Exception as e:
        raise CustomException(e,sys)