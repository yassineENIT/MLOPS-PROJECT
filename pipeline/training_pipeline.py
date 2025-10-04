import config
from config.paths_config import CONFIG_PATH, MODEL_OUTPUT_PATH, PROCESSED_DIR, PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH, TEST_FILE_PATH, TRAIN_FILE_PATH
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml

if __name__ == "__main__":
   ###1.Data ingestion
    config=read_yaml(CONFIG_PATH)
    data_ingestion=DataIngestion(config)
    data_ingestion.run()

    ### 2 data processing
    processor=DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()

    ###3 model training

    trainer=ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    trainer.run()

