import os

input_dataset = "C:\\Users\\v.seliverstov\\Documents\\isolation-data"
root_dir = os.path.dirname(os.path.abspath(__file__))

data_root_path = os.path.sep.join([root_dir, "data"])

train_path = os.path.sep.join([data_root_path, "train"])
test_path = os.path.join(data_root_path, "test")
validation_path = os.path.join(data_root_path, "validation")

train_split = 0.8
validation_split = 0.1
