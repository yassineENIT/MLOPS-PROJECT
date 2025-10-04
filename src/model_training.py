import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn



logger=get_logger(__name__)


class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path=train_path
        self.test_path=test_path
        self.model_output_path=model_output_path
        self.param_distributions=LIGHTGBM_PARAMS
        self.random_search_params=RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            # Verrouille l’ordre et l’ensemble des colonnes
            train_cols = X_train.columns.tolist()
            X_train = X_train[train_cols]
            X_test = X_test.reindex(columns=train_cols, fill_value=0)

            # Optionnel: garde un log utile
            if list(X_train.columns) != list(X_test.columns):
                only_train = [c for c in X_train.columns if c not in X_test.columns]
                only_test  = [c for c in X_test.columns if c not in X_train.columns]
                logger.warning(f"Only in train: {only_train}")
                logger.warning(f"Only in test: {only_test}")

            logger.info("Data loaded and split into features and target variable")
            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.exception("Error loading and splitting data")
            raise CustomException("Failed to load and split data", e)

            

    def train_lgbm(self,X_train,y_train):
        try:
            logger.info("Starting LightGBM model training with RandomizedSearchCV")
            lgbm = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])
            logger.info("strating random search for hyperparameter tuning")
            random_search = RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=self.param_distributions,
                n_iter=self.random_search_params["n_iter"],
                scoring=self.random_search_params["scoring"],
                cv=self.random_search_params["cv"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                n_jobs=self.random_search_params["n_jobs"]
            )
            logger.info("Fitting the model")
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            logger.info(f"Best parameters found: {random_search.best_params_}")
            return best_model
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("Failed to train model", e)
        

    def evaluate_model(self,model,X_test,y_test):
        try:
            logger.info("Evaluating the model on test data")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            logger.info(f"Model Evaluation Metrics:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
            return {

                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
             
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException("Failed to evaluate model", e)    
        
    
    def save_model(self,model):
        try:
            logger.info(f"Saving the model to {self.model_output_path}")
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving the model: {e}")
            raise CustomException("Failed to save model", e)

    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training process")

                logger.info("Starting our mlflow experimentation")

                logger.info("logging the training and testing dataset to mlflow")
                mlflow.log_artifact(self.train_path,artifact_path="datasets")
                mlflow.log_artifact(self.test_path,artifact_path="datasets")



                X_train, y_train, X_test, y_test = self.load_and_split_data()

                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)
                logger.info("logging model into mlflow")
                mlflow.log_artifact(self.model_output_path)
                logger.info("logging params and metrics to mlflow")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)
                logger.info("Model training process completed successfully")
                return metrics
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            logger.info("Model training process finished")


if __name__ =="__main__":

    trainer=ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    trainer.run()
