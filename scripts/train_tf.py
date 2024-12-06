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
import numpy as np

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

    with tf.device("/GPU:0"):
        model_hist = model.fit(x_train, y_train,
                               epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        # test_x = x_train[0].reshape(-1, x_train[0].shape[0], x_train[0].shape[1])

        memory_stats = tf.config.experimental.get_memory_info("GPU:0")
        peak_usage = round(memory_stats["peak"] / (2 ** 30), 3)
        print(f"peak memory usage: {peak_usage} GB.")

    return model

def validate_model(x_test, y_test, model):

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')



def calculate_mpjpe(predicted_poses, ground_truth_poses):
    """
    Calculates the Mean Per Joint Position Error (MPJPE) between predicted and ground truth poses.

    Args:
        predicted_poses: A numpy array of shape (N, J, 3) where N is the number of samples,
                         J is the number of joints, and 3 is for X, Y, Z coordinates.
        ground_truth_poses: A numpy array of the same shape as predicted_poses.

    Returns:
        The MPJPE value.
    """
    assert predicted_poses.shape == ground_truth_poses.shape

    error = np.linalg.norm(predicted_poses - ground_truth_poses, axis=2)
    # Todo: check if this number is better
    # error = np.mean(np.linalg.norm(predicted_poses - ground_truth_poses, dim=len(ground_truth_poses.shape) - 1))
    mpjpe = np.mean(error)

    return mpjpe


def p_mpjpe(predicted, target):
    """
    Shamelessly stolen from https://github.com/facebookresearch/VideoPose3D/blob/main/common/loss.py
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))


def calculate_models_joint_error(x=None, y=None):
    model_location = os.path.join(config.fit3d_base_directory, "mlp_model_custom_split", "keras_model_dir")

    if not x:
        x, y = get_keypoint_data(config.training_sessions, config.training_cameras,
                                 "train", 0, testing_video_names)

    # for debugging
    # x_single = x[0]
    # single_x = np.asarray(x_single)
    # single_x = preprocessing.normalize(single_x)
    # single_x = single_x.reshape(-1, single_x.shape[0], single_x.shape[1])

    keras_model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)
    predicted_poses = keras_model.predict(np.asarray(x))


    print(predicted_poses.shape)
    print(y.shape)


    predictions = np.asarray(predicted_poses)
    print(calculate_mpjpe(predictions, y))

    print(p_mpjpe(predictions, y))


def train():
    debug_videos_count = 0

    x, y = get_keypoint_data(config.training_sessions, config.training_cameras,
                             "train", debug_videos_count)

    for i, v in enumerate(x):
        x[i] = preprocessing.normalize(v)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False)
    # There is no joint3d_25 data inside of test!!!!!
    # x_test, y_test = get_keypoint_data(config.testing_sessions, config.testing_cameras, "test", debug_videos_count)

    x_train, y_train = get_keypoint_data(config.training_sessions, config.training_cameras,
                             "train", debug_videos_count, training_video_names)

    x_test, y_test = get_keypoint_data(config.training_sessions, config.training_cameras,
                             "train", debug_videos_count, testing_video_names)

    input_shape = x_train[0].shape

    model = build_mlp(input_shape)
    training_loop(x_train, y_train, model)
    print("validating mlp")
    validate_model(x_test, y_test, model)
    save_model(model, "mlp_model_custom_split")

    model = build_lstm(input_shape)
    training_loop(x_train, y_train, model)
    print("validating lstm")
    validate_model(x_test, y_test, model)
    save_model(model, "lstm_model_custom_split")



if __name__ == "__main__":
    # gpu_check()
    NUM_EPOCHS = 32
    BATCH_SIZE = 128

    training_video_names = ['overhead_extension_thruster', 'overhead_trap_raises', 'pushup', 'side_lateral_raise', 'squat',
     'standing_ab_twists', 'walk_the_box', 'warmup_1', 'warmup_10', 'warmup_11', 'warmup_12', 'warmup_13', 'warmup_14',
     'warmup_15', 'warmup_16', 'warmup_17', 'warmup_18', 'warmup_19', 'warmup_2', 'warmup_3', 'warmup_4', 'warmup_5',
     'warmup_6', 'warmup_7', 'warmup_8', 'warmup_9', 'w_raise', 'band_pull_apart', 'barbell_dead_row', 'barbell_row',
     'barbell_shrug', 'burpees', 'clean_and_press', 'deadlift', 'diamond_pushup', 'drag_curl', 'dumbbell_biceps_curls',
     'dumbbell_curl_trifecta']
    testing_video_names = ['dumbbell_hammer_curls', 'dumbbell_high_pulls', 'dumbbell_overhead_shoulder_press',
     'dumbbell_reverse_lunge', 'dumbbell_scaptions', 'man_maker', 'mule_kick', 'neutral_overhead_shoulder_press',
     'one_arm_row']

    # train()
    calculate_models_joint_error()
