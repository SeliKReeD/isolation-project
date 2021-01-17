import tensorflow
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K

# Class for building neural net model.
class CancerNet:
    @staticmethod
    def build(width,height,depth,classes):
        same_padding = "same"
        shape=(height,width,depth)
        channelDim=-1

        if K.image_data_format()=="channels_first":
            shape=(depth,height,width)
            channelDim=1

        model=Sequential([
            SeparableConv2D(32, (3,3), padding = same_padding, input_shape = shape),
            Activation(activations.relu),
            BatchNormalization(axis = channelDim),
            MaxPooling2D(pool_size = (2,2)),
            Dropout(0.25),

            SeparableConv2D(64, (3,3), padding = same_padding),
            Activation(activations.relu),
            BatchNormalization(axis = channelDim),

            SeparableConv2D(64, (3,3), padding = same_padding),
            Activation(activations.relu),
            BatchNormalization(axis = channelDim),
            MaxPooling2D(pool_size = (2,2)),
            Dropout(0.25),

            SeparableConv2D(128, (3,3), padding = same_padding),
            Activation(activations.relu),
            BatchNormalization(axis = channelDim),

            SeparableConv2D(128, (3,3), padding = same_padding),
            Activation(activations.relu),
            BatchNormalization(axis = channelDim),

            SeparableConv2D(128, (3,3), padding = same_padding),
            Activation(activations.relu),
            BatchNormalization(axis = channelDim),
            MaxPooling2D(pool_size = (2,2)),
            Dropout(0.25),

            Flatten(),
            Dense(256),
            Activation(activations.relu),
            BatchNormalization(),
            Dropout(0.5),

            Dense(classes),
            Activation(activations.softmax)
        ])

        return model
