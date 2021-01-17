import os
import config
import logging
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adagrad
from model import CancerNet
from imutils import paths
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# Initialize values for training process.
num_epochs=40
learning_rate=1e-2
batch_size=32

# Load paths to training samples.
logging.info("Start retrieving train paths.")
train_paths = list(paths.list_images(config.train_path))

# Get number of training samples.
len_train_paths = len(train_paths)
logging.info(f"Retrieved {len_train_paths} paths to training samples.")

# Get number of validation samples.
logging.info("Start retrieving validation samples paths.")
len_validation_paths = len(list(paths.list_images(config.validation_path)))
logging.info(f"Retrieved {len_validation_paths} paths to validation samples.")

# Get number of testing samples.
logging.info("Start retrieving test samples paths.")
len_test_paths = len(list(paths.list_images(config.test_path)))
logging.info(f"Retrieved {len_test_paths} paths to testing samples.")

# Generate labels for training by reading specific character in filename.
train_labels = [int(p.split(os.path.sep)[-2]) for p in train_paths]
# Convert labels into one-hot encoding format.
train_labels = to_categorical(train_labels)
# Compute sum of each class in dataset.
class_totals = train_labels.sum(axis=0)
# Compute weight for each class presented in dataset.
class_weight = class_totals.max()/class_totals
class_weight = {i : class_weight[i] for i in range(2)}
# Data augmentation for training set.
train_data_augmentation = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest")

# We dont need augmented examples in validation set,
# so we only rescale all samples.
validation_data_augmentation = ImageDataGenerator(
    rescale=1 / 255.0)

# Data generators for train, validation and test sets.
train_generator = train_data_augmentation.flow_from_directory(
    config.train_path,
    class_mode="categorical",
    target_size=(48,48),
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size)

val_generator = validation_data_augmentation.flow_from_directory(
    config.validation_path,
    class_mode="categorical",
    target_size=(48,48),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size)

test_generator = validation_data_augmentation.flow_from_directory(
    config.test_path,
    class_mode="categorical",
    target_size=(48,48),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size)

model=CancerNet.build(width = 48, height = 48,depth = 3,classes = 2)
adagrad_optimizer=Adagrad(lr = learning_rate, decay = learning_rate/num_epochs)
model.compile(
        loss = "binary_crossentropy",
        optimizer = adagrad_optimizer,
        metrics = ["accuracy"])

model.fit_generator(
  train_generator,
  steps_per_epoch=len_train_paths//batch_size,
  validation_data=val_generator,
  validation_steps=len_validation_paths//batch_size,
  class_weight=class_weight,
  epochs=num_epochs)

model_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-SNAPSHOT") + ".h5"

model.save(f"{config.root_dir}/snapshots/{model_name}")
