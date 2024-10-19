import cv2
import os


video_path = "../videos/heimburg_1.mp4"
video = cv2.VideoCapture(video_path)
out_path = "../images/" + os.path.basename(video_path).split(".")[0]

assert os.path.exists(video_path), "Could not find video path"

if not os.path.exists(out_path):
    os.makedirs(out_path)

count = 0
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

while video.isOpened():
    ret, frame = video.read()
    if ret:
        image_name = os.path.join(out_path, str(count).zfill(4) + ".jpg")
        cv2.imwrite(image_name, frame)
        count += 1
        print(f"processed {count}/{num_frames} images")
    else:
        break

video.release()
cv2.destroyAllWindows()

print("output images saved to ", out_path)
