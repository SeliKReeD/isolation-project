import config
from imutils import paths
import random, shutil, os

print(f"Start scanning for images in input folder {config.input_dataset}")
all_filepaths = list(paths.list_images(config.input_dataset))
random.seed(42)
random.shuffle(all_filepaths)

print("Split all data int train/test lists.")
train_index = int(len(all_filepaths)*config.train_split)
train_paths = all_filepaths[:train_index]
test_paths = all_filepaths[train_index:]
print(len(train_paths))

validation_index = int(len(train_paths)*config.validation_split)
validation_paths = train_paths[validation_index:]
train_paths = train_paths[:validation_index]

datasets = ([
    ("train", train_paths, config.train_path),
    ("test", test_paths, config.test_path),
    ("validation", validation_paths, config.validation_path)
])

for (type, file_paths, base_folder) in datasets:
    print(f"Start building {type} dataset.")

    if not os.path.exists(config.data_root_path):
        print(f"Creating folder {config.data_root_path}.")
        os.mkdir(config.data_root_path)

    if not os.path.exists(base_folder):
        print(f"Creating folder {base_folder}.")
        os.mkdir(base_folder)

    for file_path in file_paths:
        file_name=file_path.split(os.path.sep)[-1]
        label=file_name[-5:-4]

        labeled_path = os.path.sep.join([base_folder, label])

        if not os.path.exists(labeled_path):
            print(f"Create labeled folder {labeled_path}.")
            os.mkdir(labeled_path)

        new_path = os.path.sep.join([labeled_path, file_name])
        shutil.copy2(file_path, new_path)
