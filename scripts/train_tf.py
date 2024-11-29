import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Dropout, Conv2D, MaxPooling2D, CategoryEncoding,
                                     LSTM, ConvLSTM2D, TimeDistributed, BatchNormalization, Conv3D, Reshape, Input,
                                     MultiHeadAttention)
from data_formatter import get_keypoint_data
import config
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os

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


def save_model(model, model_name):
    """
    Converting to run on windows is tricky. Save off in a few different approaches.
    :param model: keras model
    :return:
    """
    # Saves model in directory for windows.
    tf.keras.saving.save_model(model,
                               os.path.join(config.fit3d_base_directory, model_name, "keras_model_dir"),
                               overwrite=True)
    # save model in Keras native format.
    tf.keras.saving.save_model(model,
                               os.path.join(config.fit3d_base_directory, model_name, "my_model.keras"),
                               overwrite=True)


def build_mlp(input_shape):
    """
    MLP model for (133, 3). 133 keypoints and [x, y, depth] data structure
    Output to the joints3d_25 output of (25, 3)
    """
    new_model = Sequential([
        Input(shape=input_shape),
        # LSTM(128),
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

def build_lstm(input_shape):
    """
    MLP model for (133, 3). 133 keypoints and [x, y, depth] data structure
    Output to the joints3d_25 output of (25, 3)
    """
    new_model = Sequential([
        Input(shape=input_shape),
        LSTM(128),
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

    memory_stats = tf.config.experimental.get_memory_info("GPU:0")
    peak_usage = round(memory_stats["peak"] / (2 ** 30), 3)
    print(f"peak memory usage: {peak_usage} GB.")

    # print(model.predict(test_x))
    # print(test_x)

    return model

def validate_model(x_test, y_test, model):

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')

def train():
    debug_videos_count = 12

    x, y = get_keypoint_data(config.training_sessions, config.training_cameras,
                             "train", debug_videos_count)

    for i, v in enumerate(x):
        x[i] = preprocessing.normalize(v)

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False)
    # There is no joint3d_25 data inside of test!!!!!
    # x_test, y_test = get_keypoint_data(config.testing_sessions, config.testing_cameras, "test", debug_videos_count)

    input_shape = x_train[0].shape

    model = build_mlp(input_shape)
    training_loop(x_train, y_train, model)
    print("validating mlp")
    validate_model(x_test, y_test, model)
    save_model(model, "mlp_model")

    model = build_lstm(input_shape)
    training_loop(x_train, y_train, model)
    print("validating lstm")
    validate_model(x_test, y_test, model)
    save_model(model, "mlp_model")


if __name__ == "__main__":
    # gpu_check()
    NUM_EPOCHS = 32
    BATCH_SIZE = 128
    train()
