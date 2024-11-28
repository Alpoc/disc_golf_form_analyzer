
import os
import json
import glob
import numpy as np

import config


def average_neighbors(array, x, y):
  """
  Averages the 9 points surrounding a given point in a 2D array.

  Args:
    array: The 2D NumPy array.
    x: The row index of the target point.
    y: The column index of the target point.

  Returns:
    The average value of the 9 points, or the original value if the point is on the edge.
  """

  # Determine the valid indices within the array's bounds
  min_x = max(0, round(x)-1)
  max_x = min(array.shape[0]-1, round(x)+1)
  min_y = max(0, round(y)-1)
  max_y = min(array.shape[1]-1, round(y)+1)

  # Extract the 3x3 neighborhood
  neighborhood = array[min_x:max_x+1, min_y:max_y+1]

  # Calculate the average, excluding any invalid values (e.g., points outside the array)
  return np.nanmean(neighborhood)


def get_data_for_pose_single_depth(base_path, session_number, camera, video_name):
    """
    This data loader loads input as sapiens pose x, y and for the z value it's the average around the (x, y).
    I do not expect this to work well. It's an experiment to see if it's at all successful
    """
    numpy_files = glob.glob(
        os.path.join(base_path, session_number, "depth", camera, video_name, "sapiens_0.6b", "*.npy"))
    if not len(numpy_files):
        print(f"No depth files for {video_name}")
        return [], []

    # fit3d_truth_data in joints3d_25 format instead of smplx
    joint_3d_truth_data_path = os.path.join(base_path, session_number, "joints3d_25", video_name + ".json")
    with open(joint_3d_truth_data_path) as jf:
        joint_3d_truth_data = json.load(jf)['joints3d_25']

    # sapiens pose data
    pose_jsons = glob.glob(os.path.join(base_path, session_number, "pose", camera, video_name, "sapiens_1b", "*.json"))
    pose_jsons.sort()
    poses = []
    for pose_file in pose_jsons:
        with open(pose_file) as sp:
            pose_json = json.load(sp)
        poses.append(pose_json['instance_info'][0]['keypoints'])

    # depth data from sapiens
    numpy_files = glob.glob(os.path.join(base_path, session_number, "depth", camera, video_name, "sapiens_0.6b", "*.npy"))
    numpy_files.sort()
    if not len(numpy_files):
         print(f"No depth files for {video_name}")
         return [], []
    for i, numpy_file in enumerate(numpy_files):
        depth_frame = np.load(numpy_file)
        for keypoint in poses[i]:
            x = keypoint[0]
            y = keypoint[1]
            depth_at_keypoint_location = average_neighbors(depth_frame, x, y)
            # keypoint.append(depth_at_keypoint_location)
            # normalize data
            # keypoint = [x / 1000, y / 1000, depth_at_keypoint_location]
    # check to see if the data is corrupt?
    try:
        if len(poses) and len(joint_3d_truth_data):
            print(f"{video_name}, pose: {len(poses)}, truth: {len(joint_3d_truth_data)}")
            return np.asarray(poses), np.asarray(joint_3d_truth_data)
        else:
            # TODO: this shouldn't be needed once processing all of depth is working
            return [], []
    except ValueError:
        print(f"Something is wrong with {video_name}")
        return [], []


def align_keypoints(pose_array, joint_truth):
    """
     smplx
     0 = Nose, 1 = Neck, 2 = RShoulder, 3 = RElbow, 4 = RWrist, 5 = LShoulder, 6 = LElbow, 7 = LWrist, 8 = MidHip,
     9 = RHip, 10 = RKnee, 11 = RAnkle, 12 = LHip, 13 = LKnee, 14 = LAnkle, 15 = REye, 16 = LEye, 17 = REar, 18 = LEar,
     19 = LBigToe, 20 = LSmallToe, 21 = LHeel, 22 = RBigToe, 23 = RSmallToe, 24 = RHeel. R = right, L = left.

    """
    pass


def align_data(pose_array, joint_truth):
    """
    TODO: Dont do this. Find out what is wrong with vid_to_jpg because it's output more images than truth
    There are more pictures than there is truth points
    """
    while len(joint_truth) < len(pose_array):
        pose_array = pose_array[:-1]
    return pose_array


def get_keypoint_data(sessions, cameras, mode="train", debug_amount=0):
    """
    sessions: list of session strings
    cameras: list of camera numbers
    base_path: string path to the fit3d dataset
    debug_amount: number of video frames to process for debugging. If not included process all videos
    This formater gets the data by taking the average around each keypoint location in the depth image at the
    pose keypoint locations.
    """

    base_path = os.path.join(config.fit3d_base_directory, mode)
    x_combined = []
    y_hats = []
    array_check = []
    videos_processed = 0


    for session in sessions:
        for camera in cameras:
            camera_path = os.path.join(base_path, session, "pictures", camera)
            video_names = os.listdir(camera_path)
            for i, video_name in enumerate(video_names):
                x_train, y_hat = get_data_for_pose_single_depth(base_path, session, camera, video_name)
                x_train = align_data(x_train, y_hat)

                if len(x_train):
                    videos_processed += 1

                try:
                    array_check.extend(x_train)
                    x_train = np.asarray(x_train)

                    x_combined.extend(x_train)
                    y_hats.extend(y_hat)

                    temp = np.asarray(array_check)
                    temp_2 = np.asarray(y_hats)

                    print(f"combined data: {temp.shape}, {temp_2.shape} \n")
                except Exception as exc:
                    print(f"something wrong with {video_name}")
                    print(exc)
                    continue
                # Debugging step to load less data
                if videos_processed == debug_amount:
                    break
    x_combined = np.asarray(x_combined)
    y_hats = np.asarray(y_hats)
    return x_combined, y_hats


if __name__ == '__main__':
    x, y = get_keypoint_data()
    x = align_data(x, y)
    print("loaded data")