import os 
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class Model_TrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.Model_Trainer_Config = Model_TrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Train and Test input data")
            
            X_train = train_array[:,:-1]
            Y_train = train_array[:,-1]
            
            X_test = test_array[:,:-1]
            Y_test = test_array[:,-1]
            
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }
            
            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [3, 5, 7, 10, None]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, None]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5]
                },
                "Linear Regression": {},
                "XGBoost": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7]
                },
                "CatBoost": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [50, 100, 200]
                },
                "AdaBoost": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200]
                }
            }
            
            model_report,model_params_report = evaluate_model(
                X_train = X_train,
                Y_train = Y_train,
                X_test = X_test,
                Y_test = Y_test,
                models = models,
                params=params
            )
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            best_model_params = model_params_report[best_model_name]
            
            logging.info(f"Best model :{best_model_name} & best params:{best_model_params} with scoore:{best_model_score}")
            
            if(best_model_score<0.6):
                raise CustomException("no best model found",sys)
            
            save_object(file_path=self.Model_Trainer_Config.trained_model_file_path,obj=best_model)
            
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(Y_test,predicted)
            
            return r2_square    
        
        except Exception as e:
            raise CustomException(e,sys)
        