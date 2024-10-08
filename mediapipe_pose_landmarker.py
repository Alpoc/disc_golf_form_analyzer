from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


def create_pose_land_marker():
    """Create an PoseLandmarker object."""
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.VIDEO)
    return vision.PoseLandmarker.create_from_options(options)



IMAGE_FILE = "videos/hemburg_1.mp4"
input_video = cv2.VideoCapture(IMAGE_FILE)
video_fps = input_video.get(cv2.CAP_PROP_FPS) # CAP_PROP_FPS = 5
print(video_fps)
detector = create_pose_land_marker()

while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # print(mp_image)
    timestamp_ms = int(time.time() * 1000)
    pose_landmarker_result = detector.detect_for_video(mp_image, timestamp_ms)

    print(pose_landmarker_result.pose_world_landmarks)

    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
    # cv2.imshow("", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("", annotated_image)

    if cv2.waitKey(33) == ord('p'):
        while True:
            if cv2.waitKey(33) == ord('p'):
                print("unpausing")
                break