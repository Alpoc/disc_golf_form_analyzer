import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Dropout, Conv2D, MaxPooling2D, CategoryEncoding,
                                     LSTM, ConvLSTM2D, TimeDistributed, BatchNormalization, Conv3D, Reshape, Input)
import time
from data_formatter import get_keypoint_data


def gpu_check():
    """
    Simple method to check if the GPU is available
    """
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    # print("Num GPUs Available: ", num_gpus)
    # print(tf.test.is_built_with_cuda())
    if not num_gpus:
        print('only cpu training available. Remove gpu check to continue')
        exit()
    else:
        print("GPU available")


def build_mlp(input_shape):
    """
    MLP model for (133, 3). 133 keypoints and [x, y, depth] data structure
    Output to the joints3d_25 output of (25, 3)
    """
    new_model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),

        # reshape to match the output of (25, 3) (key points, [x, y z]).
        Dense(75, activation='linear'),
        Reshape((25,3))
    ])
    new_model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])
    return new_model


def training_loop(x_train, y_train, model):

    # with tf.device("/GPU:0"):
    model_hist = model.fit(x_train, y_train,
                           epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    test_x = x_train[0].reshape(-1, x_train[0].shape[0], x_train[0].shape[1])
    print(model.predict(test_x))
    # print(test_x)

if __name__ == "__main__":
    # gpu_check()
    NUM_EPOCHS = 4
    BATCH_SIZE = 64

    x_train, y_train = get_keypoint_data()
    input_shape = x_train[0].shape
    print(f"input shape: {input_shape}")
    model = build_mlp(input_shape)
    training_loop(x_train, y_train, model)