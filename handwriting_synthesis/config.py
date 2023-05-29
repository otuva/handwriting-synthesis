import os

BASE_PATH = "model"
BASE_DATA_PATH = "data"

data_path: str = os.path.join(BASE_PATH, BASE_DATA_PATH)
processed_data_path: str = os.path.join(data_path, "processed")
raw_data_path: str = os.path.join(data_path, "raw")
ascii_data_path: str = os.path.join(raw_data_path, "ascii")

checkpoint_path: str = os.path.join(BASE_PATH, "checkpoint")
prediction_path: str = os.path.join(BASE_PATH, "prediction")
style_path: str = os.path.join(BASE_PATH, "style")
