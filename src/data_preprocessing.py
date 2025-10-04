import os 
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger=get_logger(__name__)

class DataProcessor:

    def __init__(self,train_path,test_path,processed_dir,config_path):
        
        self.train_path=train_path
        self.test_path=test_path
        self.processed_dir=processed_dir
        self.config_path = config_path
        self.config=read_yaml(self.config_path)["data_processing"]
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir,exist_ok=True)

    def preprocess_data(self,df):
        try:
            logger.info("Starting data preprocessing")
            logger.info("droping the  columns")
            df.drop(columns=['Unnamed: 0','Booking_ID'],inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols=self.config["categorical_columns"]
            num_cols=self.config["numerical_columns"]
            logger.info("Encoding categorical columns")
            label_encoder=LabelEncoder()
            mappings={}
            for col in cat_cols:
                df[col]=label_encoder.fit_transform(df[col])
                mappings[col]={label:code for label,code in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))}

            logger.info("label mappings are :{mappings}")

            logger.info("Doing skewness handling")
            skewness_threshold=self.config["skewness_threshold"]
            skewness=df[num_cols].apply(lambda x: x.skew())
            for column in skewness[skewness>skewness_threshold].index:
                df[column]=np.log1p(df[column])

            return df
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise CustomException("Error in data preprocessing",e)
        
        
    def balance_data(self,df):
        try:
            logger.info("handling data imbalance using SMOTE")
            X=df.drop(columns=['booking_status'])
            y=df['booking_status']    
            smote=SMOTE(random_state=42)
            X_resampled,y_resampled=smote.fit_resample(X,y)
            balanced_df=pd.concat([X_resampled,y_resampled],axis=1)
            logger.info("Data imbalance handled successfully")
            return balanced_df
        except Exception as e:
            logger.error(f"Error in balancing data: {e}")
            raise CustomException("Error in balancing data",e)
        

    def select_features(self,df):
        try:
            logger.info("Starting feature selection using RandomForestClassifier")
            X=df.drop(columns=['booking_status'])
            y=df['booking_status']
            model=RandomForestClassifier(random_state=42)
            model.fit(X,y)
            feature_importance=model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
            feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)  # trie sur place
            top_features_importance_df = feature_importance_df 
            num_features_to_select = self.config.get("number_of_features", self.config.get("number_of_featrues", 10))   
            top_10_features = top_features_importance_df['Feature'].head(num_features_to_select).values
            top_10_df = df[top_10_features.tolist() + ['booking_status']]

            logger.info(f"Top {num_features_to_select} features selected: {top_10_features}")
            return top_10_df
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise CustomException("Error in feature selection",e)
        

    def save_data(self,df,file_path):
        try:
            logger.info(f"Saving data to {file_path}")
            df.to_csv(file_path,index=False)
            logger.info(f"Data saved successfully at {file_path}")
        except Exception as e:
            logger.error(f"Error in saving data: {e}")
            raise CustomException("Error in saving data",e)
    
    def process(self):
        try:
            logger.info("Loading training data")
            train_df=load_data(self.train_path)
            logger.info("Loading testing data")
            test_df=load_data(self.test_path)

            logger.info("Preprocessing training data")
            train_df=self.preprocess_data(train_df)
            logger.info("Preprocessing testing data")
            test_df=self.preprocess_data(test_df)

            logger.info("Balancing training data")
            train_df=self.balance_data(train_df)
            logger.info("Balancing testing data")
            test_df=self.balance_data(test_df)

            logger.info("Selecting features from training data")
            train_df=self.select_features(train_df)

            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully")

        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise CustomException("Error in data processing",e)
        


if __name__=="__main__":
    processor=DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()



