import cv2
import matplotlib.pyplot as plt
import json
import matplotlib
import numpy as np
from dataset_util import read_cam_params, project_3d_to_2d, plot_over_image

import plotly.graph_objects as go

def show3Dpose(channels, ax, radius=.5, mpii=2, lcolor='#ff0000', rcolor='#0000ff'):
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

    plt.savefig("2d.jpg")

def plot_2d(plot_data):
    # joints_2d = project_3d_to_2d(plot_data, cam_params['intrinsics_wo_distortion'], 'wo_distortion')
    ax = plt.figure().add_subplot(projection='3d')
    show3Dpose(plot_data, ax)


def plot_plotly(joint_data):
    fig = go.Figure(data=[go.Scatter3d(x=joint_data[:, 0], y=joint_data[:, 1], z=joint_data[:, 2], mode='markers+lines')])
    fig.show()

def plot_3d(plot_data):
    ax = plt.figure().add_subplot(projection='3d')

    for joint in plot_data:
        ax.scatter(joint[0], joint[1], joint[2])
    plt.savefig("plot.png")


def annotate_image():
    image = cv2.imread("/media/dj/3CB88F62B88F1992/fit3d/s03/pictures/50591643/band_pull_apart/0000.jpg")
    center = (461, 199)
    annotated_image = cv2.circle(image, center, radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imshow("annotated", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


with open("/media/dj/3CB88F62B88F1992/fit3d/s03/joints3d_25/band_pull_apart.json") as json_file:
    data = json.load(json_file)

cam_params = read_cam_params("/media/dj/3CB88F62B88F1992/fit3d/s03/camera_parameters/50591643/band_pull_apart.json")



body_points = data["joints3d_25"]
print(body_points[0])
# plot_3d(body_points[0])
plot_2d(np.asarray(body_points[0]))
# plot_plotly(np.asarray(body_points[0]))