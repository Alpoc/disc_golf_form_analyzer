import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from dataset_util import read_cam_params, project_3d_to_2d, plot_over_image

# import plotly.graph_objects as go
import tensorflow as tf
from sklearn import preprocessing
import config
import os

from data_formatter import get_keypoint_data


def show3Dpose(channels, ax, radius=.5, mpii=2, lcolor='#ff0000', rcolor='#0000ff'):
    """
    Taken from fit3d code https://github.com/sminchisescu-research/imar_vision_datasets_tools/blob/main/util/dataset_util.py
    """
    vals = channels

    if mpii == 0: # h36m with mpii joints
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    elif mpii == 1: # only mpii
        connections = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                       [7, 8], [8, 9], [7, 12], [12, 11], [11, 10], [7, 13], [13, 14], [14, 15]]
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    else: # default h36m
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    for ind, (i,j) in enumerate(connections):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[ind] else rcolor)

    RADIUS = radius  # space around the subject
    if mpii == 1:
        xroot, yroot, zroot = vals[6, 0], vals[6, 1], vals[6, 2]
    else:
        xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return plt
    # plt.savefig("2d_truth.jpg")


def plot_over_image(frame, points_2d=np.array([]), with_ids=True, with_limbs=True, path_to_write=None):
    """
    also taken from https://github.com/sminchisescu-research/imar_vision_datasets_tools/blob/main/util/dataset_util.py
    """
    num_points = points_2d.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(frame)
    if points_2d.shape[0]:
        ax.plot(points_2d[:, 0], points_2d[:, 1], 'x', markeredgewidth=10, color='white')
        if with_ids:
            for i in range(num_points):
                ax.text(points_2d[i, 0], points_2d[i, 1], str(i), color='red', fontsize=20)
        if with_limbs:
            limbs = [[10, 9], [9, 8], [8, 11], [8, 14], [11, 12], [14, 15], [12, 13], [15, 16],
                     [8, 7], [7, 0], [0, 1], [0, 4], [1, 2], [4, 5], [2, 3], [5, 6],
                     [13, 21], [13, 22], [16, 23], [16, 24], [3, 17], [3, 18], [6, 19], [6, 20]]
            for limb in limbs:
                if limb[0] < num_points and limb[1] < num_points:
                    ax.plot([points_2d[limb[0], 0], points_2d[limb[1], 0]],
                            [points_2d[limb[0], 1], points_2d[limb[1], 1]],
                            linewidth=12.0)

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if path_to_write:
        plt.ioff()
        plt.savefig(path_to_write, pad_inches=0, bbox_inches='tight')


def plot_2d(plot_data):
    # joints_2d = project_3d_to_2d(plot_data, cam_params['intrinsics_wo_distortion'], 'wo_distortion')
    ax = plt.figure().add_subplot(projection='3d')
    plott = show3Dpose(plot_data, ax)
    return plott


def annotate_image():
    image = cv2.imread("/media/dj/3CB88F62B88F1992/fit3d/s03/pictures/50591643/band_pull_apart/0000.jpg")
    center = (461, 199)
    annotated_image = cv2.circle(image, center, radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imshow("annotated", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_truth():
    """
    Visualize and save to 2d_truth.jpg
    """
    with open("/media/dj/3CB88F62B88F1992/fit3d/train/s03/joints3d_25/overhead_extension_thruster.json") as json_file:
        data = json.load(json_file)

    body_points = data["joints3d_25"]
    for i, body_point in enumerate(body_points):
        plotty = plot_2d(np.asarray(body_point))
        plotty.savefig("../images/truth_plots/" + str(i).zfill(4) + ".jpg")


def visualize_from_model():
    model_location = os.path.join(config.fit3d_base_directory, "mlp_model", "keras_model_dir")

    x, y = get_keypoint_data(config.training_sessions, config.training_cameras,
                             "train", 1)
    for i, x_single in enumerate(x):
        single_x = np.asarray(x_single)
        single_x = preprocessing.normalize(single_x)
        single_x = single_x.reshape(-1, single_x.shape[0], single_x.shape[1])

        keras_model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)
        key_points = keras_model.predict(single_x)[0]
        # print(key_points)
        plotty = plot_2d(key_points)
        plotty.show()
        # plotty.savefig("../images/predictions" + str(i).zfill(4) + ".jpg")


if __name__ == "__main__":
    visualize_truth()
    # visualize_from_model()